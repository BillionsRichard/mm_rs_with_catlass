#ifndef CATCOC_DGEMM_KERNEL_MATMUL_ALLREDUCE_HPP
#define CATCOC_DGEMM_KERNEL_MATMUL_ALLREDUCE_HPP

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
    class BlockEpilogueReduceScatter_,
    class BlockEpilogueAllGather_,
    class BlockScheduler_,
    class BlockEpilogueScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class MatmulAllReduce {
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

    using ReduceScatter = BlockEpilogueReduceScatter_;
    using ReduceScatterParams = typename ReduceScatter::Params;

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

        uint32_t rankIdx;
        uint32_t rankSize;

        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrSymmetric;
        ReduceScatterParams reduceScatterParams;
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
            uint32_t commInterval_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrD_, LayoutD const &layoutD_,
            GM_ADDR ptrSymmetric_,
            ReduceScatterParams const &reduceScatterParams_,
            AllGatherParams const &allGatherParams_
        ) : problemShape(problemShape_),
            rankIdx(rank_), rankSize(rankSize_),
            commInterval(commInterval_),
            ptrA(ptrA_), layoutA(layoutA_),
            ptrB(ptrB_), layoutB(layoutB_),
            ptrD(ptrD_), layoutD(layoutD_),
            ptrSymmetric(ptrSymmetric_),
            reduceScatterParams(reduceScatterParams_),
            allGatherParams(allGatherParams_) {}
    };

    // Methods
    CATLASS_DEVICE
    MatmulAllReduce()
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
        BlockScheduler matmulBlockScheduler(params.problemShape, blockShape.GetCoordMN());
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));

        // Comm need repeat
        uint32_t aicoreIndex = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        uint32_t blockPerComm = aicoreNum * params.commInterval;
        uint32_t commLoops = CeilDiv(coreLoops, blockPerComm);

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric));

        auto layoutC = Catlass::layout::RowMajor{
            WORKSPACE_STAGES * blockPerComm * L1TileShape::M, L1TileShape::N,
            L1TileShape::N
        };

        auto layoutCRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, blockPerComm, L1TileShape::M);
        auto layoutCRow = layout::AffineRankN<3>::Packed(layoutCRowLogicShape);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;
            
            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageId]);
            }

            uint32_t commBlockOffset = commIdx * blockPerComm;
            for (
                uint32_t blockIdxInComm = aicoreIndex, loopIdx = commBlockOffset + aicoreIndex;
                blockIdxInComm < blockPerComm && loopIdx < coreLoops;
                blockIdxInComm += aicoreNum, loopIdx = commBlockOffset + blockIdxInComm
            ) {
                // Compute block location
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                GemmCoord offsetCoord = blockCoord * blockShape;
                // Compute initial location in logical coordinates
                auto blockOffsetA = offsetCoord.GetCoordMK();
                auto blockOffsetB = offsetCoord.GetCoordKN();
                MatrixCoord blockOffsetC{layoutCRow(Catlass::MakeCoord<int>(stageId, blockIdxInComm, 0)), 0};

                int64_t offsetA = params.layoutA.GetOffset(blockOffsetA);
                int64_t offsetB = params.layoutB.GetOffset(blockOffsetB);
                int64_t offsetC = layoutC.GetOffset(blockOffsetC);

                // Compute block-scoped matrix multiply-add
                blockMmad(
                    gmA[offsetA], params.layoutA,
                    gmB[offsetB], params.layoutB,
                    gmC[offsetC], layoutC,
                    actualBlockShape
                );
            }

            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageId]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params &params)
    {
        MatrixCoord blockShapeMN = L1TileShape::ToCoordMN();
        BlockScheduler matmulBlockScheduler(params.problemShape, blockShapeMN);
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        ReduceScatter reduceScatter(resource, params.reduceScatterParams);
        AllGather allGather(resource, params.allGatherParams);

        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aivIndex = AscendC::GetSubBlockIdx();

        auto blockPerComm = aicoreNum * params.commInterval;
        auto commLoops = CeilDiv(coreLoops, blockPerComm);

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric));

        auto layoutC = Catlass::layout::RowMajor{
            WORKSPACE_STAGES * blockPerComm * L1TileShape::M, L1TileShape::N,
            L1TileShape::N
        };

        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD *>(params.ptrD));

        MatrixCoord commBlockShape = params.reduceScatterParams.BlockShape();
        MatrixCoord commCoreSplit = params.reduceScatterParams.CoreSplit();
        CommScheduler commScheduler(params.rankIdx, params.rankSize, commBlockShape, commCoreSplit);
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;
            uint32_t actualBlockInComm = Min(blockPerComm, coreLoops - commIdx * blockPerComm);
            MatrixCoord actualCommShape = MatrixCoord{actualBlockInComm, 1} * blockShapeMN;


            commScheduler.template SetProblemSize<ReduceScatter::RemoteCopyMode, true>(actualCommShape);
            auto commAicoreNum = commScheduler.GetRealCore();
            auto commCoreLoops = commScheduler.GetCoreLoop();

            MatrixCoord stageOffset = MatrixCoord{stageId * blockPerComm, 0} * blockShapeMN;
            MatrixCoord commOffset = MatrixCoord{commIdx * blockPerComm, 0} * blockShapeMN;

            // wait aic
            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageId]);

            // Local matmul is completed, waiting until tasks on all devices are complete.
            shmemx_barrier_all_vec();

            AscendC::SetAtomicAdd<ElementD>();
            AscendC::PipeBarrier<PIPE_ALL>();
            reduceScatter.AllocEventID();
            if (aivIndex == 0 && aicoreIndex < commAicoreNum) {
                for (uint32_t commLoopIdx = aicoreIndex; commLoopIdx < commCoreLoops; commLoopIdx += commAicoreNum) {
                    MatrixCoord commBlockCoord = commScheduler.GetBlockIdx(commLoopIdx);
                    MatrixCoord blockOffset = commScheduler.template GetBlockOffset<ReduceScatter::RemoteCopyMode,
                        ReduceScatter::RemoteCopyDirect>(commBlockCoord);
                    MatrixCoord actualCommBlockShape = commScheduler.template GetActualBlockShape<
                        ReduceScatter::RemoteCopyMode, ReduceScatter::RemoteCopyDirect>(commBlockCoord);

                    uint32_t remoteRankIdx = commBlockCoord.column();
                    if (remoteRankIdx == params.rankIdx) {
                        continue;
                    }

                    auto offsetIn = stageOffset + blockOffset;
                    auto offsetOut = offsetIn;

                    auto globalLoopIdx = (commOffset + blockOffset).row() / blockShapeMN.row();

                    reduceScatter(blockShapeMN, offsetOut, offsetIn, actualCommBlockShape,
                        gmC, layoutC, globalLoopIdx, remoteRankIdx % params.rankSize);
                }
            }
            reduceScatter.ReleaseEventID();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::SetAtomicNone();
            AscendC::PipeBarrier<PIPE_ALL>();

            // ReduceScatter is completed, waiting until tasks on all devices are complete.
            shmemx_barrier_all_vec();

            allGather.AllocEventID();
            if (aivIndex == 0 && aicoreIndex < commAicoreNum) {
                for (uint32_t commLoopIdx = aicoreIndex; commLoopIdx < commCoreLoops; commLoopIdx += commAicoreNum) {
                    MatrixCoord commBlockCoord = commScheduler.GetBlockIdx(commLoopIdx);
                    MatrixCoord blockOffset = commScheduler.template GetBlockOffset<AllGather::RemoteCopyMode,
                        AllGather::RemoteCopyDirect>(commBlockCoord);
                    MatrixCoord actualCommBlockShape = commScheduler.template GetActualBlockShape<
                        AllGather::RemoteCopyMode, AllGather::RemoteCopyDirect>(commBlockCoord);

                    uint32_t remoteRankIdx = commBlockCoord.column();

                    auto offsetIn = stageOffset + blockOffset;
                    auto offsetOut = commOffset + blockOffset;

                    auto globalLoopIdx = offsetOut.row() / blockShapeMN.row();

                    allGather(blockShapeMN, offsetOut, offsetIn, actualCommBlockShape,
                        gmD, params.layoutD, globalLoopIdx, remoteRankIdx % params.rankSize);
                }
            }
            allGather.ReleaseEventID();
            // AllGather is completed, waiting until tasks on all devices are complete.
            shmemx_barrier_all_vec();

            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId]);
        }
    }

private:
    // ID used for inter-core synchronization
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace Catco::DGemm::Kernel

#endif // CATCOC_DGEMM_KERNEL_MATMUL_ALLREDUCE_HPP
