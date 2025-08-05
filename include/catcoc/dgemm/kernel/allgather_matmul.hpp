#ifndef CATCOC_DGEMM_KERNEL_ALLGATHER_MATMUL_HPP
#define CATCOC_DGEMM_KERNEL_ALLGATHER_MATMUL_HPP

#include "catcoc/catcoc.hpp"

// from catlass
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catcoc::DGemm::Kernel {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

template <
    class BlockMmad_,
    class BlockEpilogueAllGather_,
    class BlockScheduler_,
    class BlockEpilogueScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class AllGatherMatmul {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using AllGather = BlockEpilogueAllGather_;
    using AllGatherParams = typename AllGather::Params;

    using ElementD = typename AllGather::ElementDst;
    using LayoutD = typename AllGather::LayoutDst;

    using BlockScheduler = BlockScheduler_;
    using CommScheduler = BlockEpilogueScheduler_;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    
    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GemmCoord blockShape;

        uint32_t rankIdx;
        uint32_t rankSize;

        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrSymmetric;
        AllGatherParams allGatherParams;

        GM_ADDR ptrD;
        LayoutD layoutD;

        uint32_t commInterval;

        // Methods
        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(
            GemmCoord const &problemShape_,
            uint32_t rank_, uint32_t rankSize_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrSymmetric_,
            AllGatherParams const &allGatherParams_,
            GM_ADDR ptrD_, LayoutD const &layoutD_,
            uint32_t commInterval_
        ) : problemShape(problemShape_),
            rankIdx(rank_), rankSize(rankSize_),
            ptrA(ptrA_), layoutA(layoutA_),
            ptrB(ptrB_), layoutB(layoutB_),
            ptrSymmetric(ptrSymmetric_),
            allGatherParams(allGatherParams_),
            ptrD(ptrD_), layoutD(layoutD_),
            commInterval(commInterval_) {}
    };

    // Methods
    CATLASS_DEVICE
    AllGatherMatmul()
    {
        for (uint32_t i = 0; i < WORKSPACE_STAGES; ++i) {
            flagAicFinishStore[i] = Catlass::Arch::CrossCoreFlag(i);
            flagAivFinishCompute[i] = Catlass::Arch::CrossCoreFlag(i);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params &params)
    {
        GemmCoord blockShape = L1TileShape::ToCoord();
        BlockScheduler matmulBlockScheduler(params.rankSize, params.commInterval, params.problemShape, blockShape.GetCoordMN());

        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrD));

        // Comm need repeat
        uint32_t aicoreIndex = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        auto mLoops = CeilDiv(matmulBlockScheduler.GetMLoops(), params.rankSize);
        auto nLoops = matmulBlockScheduler.GetNLoops();
        auto blockPerComm = params.commInterval * params.rankSize * nLoops;
        auto mLoopsPerComm = params.commInterval * params.rankSize;
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
        uint32_t commLoops = CeilDiv(coreLoops, blockPerComm);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;
            auto actualBlocksPerComm = (commIdx == commLoops - 1) ? 
                coreLoops - commIdx * blockPerComm : blockPerComm;
            auto mLoopsPerRank = actualBlocksPerComm / (params.rankSize * nLoops);

            // wait aiv
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageId]);

            for (uint32_t compIdx = aicoreIndex; compIdx < actualBlocksPerComm; compIdx += aicoreNum) {
                auto loopIdx = commIdx * blockPerComm + compIdx;

                // Compute block location
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                uint32_t rankIdx = blockCoord.m() / mLoops;
                uint32_t mIdxInRank = blockCoord.m() % mLoops;
                uint32_t inCommIdx = mIdxInRank % params.commInterval;

                auto layoutARowLogicShape = Catlass::MakeCoord<int64_t>(mLoopsPerComm, mLoopsPerRank, 1);
                auto layoutARow = layout::AffineRankN<3>(layoutARowLogicShape);

                GemmCoord offsetCoord = blockCoord * blockShape;
                MatrixCoord rankOffsetC = params.problemShape.GetCoordMN() * Catlass::MakeCoord<uint32_t>(rankIdx, 0);
                MatrixCoord inRankOffsetC = MatrixCoord{mIdxInRank, blockCoord.n()} * blockShape.GetCoordMN();

                auto blockOffsetA = MatrixCoord{(uint32_t)layoutARow(Catlass::MakeCoord<int>(stageId, rankIdx, inCommIdx)) * L1TileShape::M, offsetCoord.k()};
                auto blockOffsetB = offsetCoord.GetCoordKN();
                auto blockOffsetC = rankOffsetC + inRankOffsetC;
                int64_t offsetA = params.layoutA.GetOffset(blockOffsetA);
                int64_t offsetB = params.layoutB.GetOffset(blockOffsetB);
                int64_t offsetC = params.layoutD.GetOffset(blockOffsetC);

                // Compute block-scoped matrix multiply-add
                blockMmad(
                    gmA[offsetA], params.layoutA,
                    gmB[offsetB], params.layoutB,
                    gmC[offsetC], params.layoutD,
                    actualBlockShape);
            }

            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageId]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params &params)
    {
        MatrixCoord blockShapeMN = MatrixCoord{L1TileShape::M, params.problemShape.k()};
        BlockScheduler matmulBlockScheduler(params.rankSize, params.commInterval, params.problemShape, L1TileShape::ToCoordMN());

        AllGather allGather(resource, params.allGatherParams);
        
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aivIndex = AscendC::GetSubBlockIdx();

        // Split core loop to comm loop tile
        auto blockPerComm = params.commInterval * params.rankSize;
        uint32_t coreLoops = matmulBlockScheduler.GetMLoops();
        auto commLoops = CeilDiv(coreLoops, blockPerComm);

        AscendC::GlobalTensor<ElementA> tensorA;
        tensorA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));

        MatrixCoord commBlockShape = params.allGatherParams.BlockShape();
        MatrixCoord commCoreSplit = params.allGatherParams.CoreSplit();
        CommScheduler commScheduler(params.rankIdx, params.rankSize, commCoreSplit);
        auto commAicoreNum = commScheduler.GetRealCore();

        auto layoutCommBlockLogicShapeInRank = Catlass::MakeCoord<int>(1, 0, commBlockShape.row());
        auto layoutCommBlockInRank = layout::AffineRankN<3>::Packed(layoutCommBlockLogicShapeInRank);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;
            uint32_t actualBlockInComm = Min(blockPerComm, coreLoops - commIdx * blockPerComm);
            MatrixCoord actualCommShape = MatrixCoord{actualBlockInComm, 1} * blockShapeMN;

            MatrixCoord dataLoopsMx = CeilDiv(actualCommShape, commBlockShape);
            uint32_t dLoopsInRank = CeilDiv(dataLoopsMx.row() * dataLoopsMx.column(), params.rankSize);
            commScheduler.Update(actualCommShape, commBlockShape, dLoopsInRank);
            auto commCoreLoops = commScheduler.GetCoreLoop();

            auto layoutCommBlockLogicShape = Catlass::MakeCoord<int>(1, dLoopsInRank, commBlockShape.row());
            auto layoutCommBlock = layout::AffineRankN<3>::Packed(layoutCommBlockLogicShape);

            MatrixCoord commOffset = MatrixCoord{commIdx * params.commInterval, 0} * blockShapeMN; 
            MatrixCoord stageOffset = MatrixCoord{stageId * blockPerComm, 0} * blockShapeMN;

            // wait aic
            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageId]);
            }

            shmemx_barrier_all_vec();

            allGather.AllocEventID();
            if (aivIndex == 0 && aicoreIndex < commAicoreNum) {
                for (uint32_t commLoopIdx = aicoreIndex; commLoopIdx < commCoreLoops; commLoopIdx += commAicoreNum) {
                    MatrixCoord commBlockCoord = commScheduler.GetBlockIdx(commLoopIdx);
                    MatrixCoord inputBlockOffset = commScheduler.template GetBlockOffset<AllGather::RemoteCopyMode,
                        AllGather::RemoteCopyDirect>(commBlockCoord, layoutCommBlockInRank);
                    MatrixCoord outputBlockOffset = commScheduler.template GetBlockOffset<AllGather::RemoteCopyMode,
                        AllGather::RemoteCopyDirect>(commBlockCoord, layoutCommBlock);
                    MatrixCoord actualCommSubBlockShape = commScheduler.template GetActualBlockShape<
                        AllGather::RemoteCopyMode, AllGather::RemoteCopyDirect>(commBlockCoord, layoutCommBlock);
 
                    uint32_t remoteRankIdx = commBlockCoord.column();

                    auto offsetIn = commOffset + inputBlockOffset;
                    auto offsetOut = stageOffset + outputBlockOffset;
                    
                    MatrixCoord inputLoopOffset = offsetIn / blockShapeMN;
                    auto globalLoopIdx = inputLoopOffset.row();
 
                    allGather(blockShapeMN, offsetOut, offsetIn, actualCommSubBlockShape, 
                        tensorA, params.layoutA, globalLoopIdx, remoteRankIdx % params.rankSize);
                }
            }
            allGather.ReleaseEventID();
            // AllGather is completed, waiting until tasks on all devices are complete.
            shmemx_barrier_all_vec();

            // set aic
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId]);
        }

    }

private:
    // ID used for inter-core synchronization
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace Catcoc::Gemm::Kernel

#endif // CATCOC_DGEMM_KERNEL_ALLGATHER_MATMUL_HPP
