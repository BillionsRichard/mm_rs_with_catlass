#ifndef CATCOC_DGEMM_KERNEL_ALLGATHER_DEQUANT_MATMUL_HPP
#define CATCOC_DGEMM_KERNEL_ALLGATHER_DEQUANT_MATMUL_HPP

#include "catcoc/catcoc.hpp"

// from catlass
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catcoc::DGemm::Kernel {

using Catlass::GemmCoord;
using Catlass::MatrixCoord;

template <
    class BlockMmad_,
    class BlockEpilogueAllGather_,
    class BlockSchedulerForAllgather_,
    class BlockEpilogue_,
    class BlockSchedulerForDequant_,
    class BlockEpilogueScheduler_,
    uint32_t WORKSPACE_STAGES_>
class AllGatherDequantMatmul {
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
    using Dequant = BlockEpilogue_;

    using ElementScale = typename Dequant::ElementScale;
    using LayoutScale = typename Dequant::LayoutScale;
    using ElementPerTokenScale = typename Dequant::ElementPerTokenScale;
    using LayoutPerTokenScale = typename Dequant::LayoutPerTokenScale;
    using ElementD = typename Dequant::ElementD;
    using LayoutD = typename Dequant::LayoutD;

    using AllGatherParams = typename AllGather::Params;
    using DequantParams = typename Dequant::Params;

    using BlockSchedulerForAllgather = BlockSchedulerForAllgather_;
    using BlockSchedulerForDequant = BlockSchedulerForDequant_;
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
        GM_ADDR ptrC;
        LayoutB layoutC;
        GM_ADDR ptrSymmetric;
        AllGatherParams allGatherParams;
        DequantParams dequantParams;
        uint32_t commInterval;

        // Methods
        CATLASS_DEVICE
        Params()
        {}

        CATLASS_DEVICE
        Params(
            GemmCoord const &problemShape_,
            uint32_t rank_, uint32_t rankSize_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            GM_ADDR ptrSymmetric_,
            AllGatherParams const &allGatherParams_,
            DequantParams const &dequantParams_,
            uint32_t commInterval_
        ) : problemShape(problemShape_),
            rankIdx(rank_), rankSize(rankSize_),
            ptrA(ptrA_), layoutA(layoutA_),
            ptrB(ptrB_), layoutB(layoutB_),
            ptrC(ptrC_), layoutC(layoutC_),
            ptrSymmetric(ptrSymmetric_),
            allGatherParams(allGatherParams_),
            dequantParams(dequantParams_),
            commInterval(commInterval_)
        {}
    };

    // Methods
    CATLASS_DEVICE
    AllGatherDequantMatmul()
    {
        for (uint32_t i = 0; i < WORKSPACE_STAGES; ++i) {
            flagAicFinishStore[i] = Catlass::Arch::CrossCoreFlag(i);    // 将id设置为0,1... (WORKSPACE_STAGES-1)
            flagAivFinishCompute[i] = Catlass::Arch::CrossCoreFlag(i);
        }
        flagAicFinish = Catlass::Arch::CrossCoreFlag(WORKSPACE_STAGES);
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        GemmCoord blockShape = L1TileShape::ToCoord();
        uint32_t commSizeM = params.commInterval * L1TileShape::M;
        uint32_t commLoops = CeilDiv(params.problemShape.m(), commSizeM);

        BlockMmad mmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));

        auto layoutSymmetric = Catlass::layout::RowMajor(WORKSPACE_STAGES * params.rankSize * commSizeM,
            params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementA)));
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        auto layoutC = params.layoutC;
        auto layoutCRowLogicStride = Catlass::MakeCoord<int64_t>(params.problemShape.m(), commSizeM, 1);
        auto layoutCRow = layout::AffineRankN<3>(layoutCRowLogicStride);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            uint32_t actualCommSizeM = Min(commSizeM, params.problemShape.m() - commIdx * commSizeM);
            auto actualProblemShape = Catlass::MakeCoord<uint32_t>(
                actualCommSizeM, params.problemShape.n(), params.problemShape.k(), params.rankSize);
            BlockSchedulerForAllgather mmadScheduler(actualProblemShape, blockShape.GetCoordMN());
            uint32_t coreLoops = mmadScheduler.GetCoreLoops();

            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageId]);
            for (uint32_t loopIdx = aicoreIdx; loopIdx < coreLoops; loopIdx += aicoreNum) {
                auto blockOffset = mmadScheduler.GetBlockOffset(loopIdx);
                auto actualBlockShape = mmadScheduler.GetActualBlockShapeByOffset(blockOffset);

                uint32_t srcRankIdx = blockOffset.rank();
                MatrixCoord commOffsetA{layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, srcRankIdx, 0)), 0};
                MatrixCoord commOffsetC{layoutCRow(Catlass::MakeCoord<int>(srcRankIdx, commIdx, 0)), 0};

                MatrixCoord offsetA = commOffsetA + blockOffset.GetCoordMK();
                MatrixCoord offsetB = blockOffset.GetCoordKN();
                MatrixCoord offsetC = commOffsetC + blockOffset.GetCoordMN();

                auto gmBlockA = gmSymmetric[layoutSymmetric.GetOffset(offsetA)];
                auto gmBlockB = gmB[params.layoutB.GetOffset(offsetB)];
                auto gmBlockC = gmC[layoutC.GetOffset(offsetC)];

                mmad(gmBlockA,
                    layoutSymmetric,
                    gmBlockB,
                    params.layoutB,
                    gmBlockC,
                    layoutC,
                    actualBlockShape.GetCoordMNK());
            }
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageId]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        Catlass::Arch::CrossCoreBarrier<0, PIPE_FIX>();
        // 0x2->模式2：子块间的同步，对一个组中的所有子块设置flagId。
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinish);
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params &params)
    {
        uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t subcoreIdx = AscendC::GetSubBlockIdx();

        MatrixCoord blockShapeMK = MatrixCoord{L1TileShape::M, params.problemShape.k()};
        uint32_t commSizeM = params.commInterval * L1TileShape::M;
        uint32_t commLoops = CeilDiv(params.problemShape.m(), commSizeM);

        AllGather allGather(resource, params.allGatherParams);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
        AscendC::GlobalTensor<ElementA> gmSymmetric;
        gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric));

        auto layoutSymmetric = Catlass::layout::RowMajor(WORKSPACE_STAGES * params.rankSize * commSizeM,
            params.problemShape.k(),
            RoundUp<int64_t>(params.problemShape.k(), Catlass::BYTE_PER_FRACTAL / sizeof(ElementA)));
        auto layoutSymmetricRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, params.rankSize, commSizeM);
        auto layoutSymmetricRow = layout::AffineRankN<3>::Packed(layoutSymmetricRowLogicShape);

        MatrixCoord commBlockShape = params.allGatherParams.BlockShape();
        MatrixCoord commCoreSplit = params.allGatherParams.CoreSplit();
        CommScheduler commScheduler(commBlockShape, commCoreSplit);
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            uint32_t actualCommSizeM = Min(commSizeM, params.problemShape.m() - commIdx * commSizeM);
            auto actualCommShape = DistMatrixCoord(actualCommSizeM, params.problemShape.k(), params.rankSize);
            MatrixCoord loopsInRank = CeilDiv(MatrixCoord(actualCommShape.GetCoordInRank()), commBlockShape);
            commScheduler.UpdateProblem(actualCommShape, loopsInRank);

            auto commAicoreNum = commScheduler.GetRealCore();
            auto commCoreLoops = commScheduler.GetCoreLoop();

            MatrixCoord commSrcOffset{commIdx * commSizeM, 0};
            MatrixCoord commDstOffset{layoutSymmetricRow(Catlass::MakeCoord<int>(stageId, params.rankIdx, 0)), 0};

            // wait aic
            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageId]);
            }

            shmemx_barrier_all_vec();

            allGather.InitBlockLoop();
            if (subcoreIdx == 0 && aicoreIdx < commAicoreNum) {
                for (uint32_t commLoopIdx = aicoreIdx; commLoopIdx < commCoreLoops; commLoopIdx += commAicoreNum) {
                    DistMatrixCoord commBlockCoord = commScheduler.GetBlockCoord(commLoopIdx);
                    MatrixCoord blockOffsetInRank = commScheduler.GetBlockOffsetInRank(commBlockCoord.GetCoordInRank());
                    MatrixCoord actualCommBlockShape = commScheduler.GetActualBlockShapeByOffset(blockOffsetInRank);

                    uint32_t remoteRankIdx = commBlockCoord.rank();

                    auto offsetSrc = commSrcOffset + blockOffsetInRank;
                    auto offsetDst = commDstOffset + blockOffsetInRank;

                    auto gmBlockSrc = gmA[params.layoutA.GetOffset(offsetSrc)];
                    auto layoutBlockSrc = params.layoutA.GetTileLayout(actualCommBlockShape);

                    auto gmBlockDst = gmSymmetric[layoutSymmetric.GetOffset(offsetDst)];
                    auto layoutBlockDst = layoutSymmetric.GetTileLayout(actualCommBlockShape);

                    allGather(gmBlockSrc,
                        layoutBlockSrc,
                        gmBlockDst,
                        layoutBlockDst,
                        actualCommBlockShape,
                        remoteRankIdx % params.rankSize);
                }
            }
            allGather.FinalizeBlockLoop();
            // AllGather is completed, waiting until tasks on all devices are complete.
            shmemx_barrier_all_vec();

            // set aic
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId]);
        }

        Catlass::Arch::CrossCoreWaitFlag(flagAicFinish);
        Catlass::Arch::CrossCoreBarrier<0, PIPE_MTE3>();

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
        auto layoutC = params.layoutC;

        Dequant dequant(resource);
        for (uint32_t deviceIdx = 0; deviceIdx < params.rankSize; ++deviceIdx) {
            BlockSchedulerForDequant blockSchedulerForDequant(params.problemShape, L1TileShape::ToCoordMN());
            typename Dequant::Params paramsForDequant{params.dequantParams.ptrScale,
                params.dequantParams.layoutScale,
                params.dequantParams.ptrPerTokenScale,
                params.dequantParams.layoutPerTokenScale,
                params.dequantParams.ptrD + deviceIdx * params.problemShape.m() * params.problemShape.n(),
                params.dequantParams.layoutD};
            dequant.UpdateParams(paramsForDequant);
            uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            uint32_t coreNum = AscendC::GetBlockNum();
            uint32_t coreLoops = blockSchedulerForDequant.GetCoreLoops();
            GemmCoord blockShapeMNK = L1TileShape::ToCoord();
            for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                GemmCoord blockCoordMNK = blockSchedulerForDequant.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShapeMNK = blockSchedulerForDequant.GetActualBlockShape(blockCoordMNK);
                // MatrixCoord offsetC{blockCoordMNK.m() * L1TileShape::M, blockCoordMNK.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoordMNK.m() * L1TileShape::M, blockCoordMNK.n() * L1TileShape::N};
                int64_t gmOffsetC =
                    layoutC.GetOffset(offsetC) + deviceIdx * params.problemShape.m() * params.problemShape.n();
                auto gmBlockC = gmC[gmOffsetC];
                auto layoutBlockC = layoutC.GetTileLayout(actualBlockShapeMNK.GetCoordMN());
                dequant(blockShapeMNK, blockCoordMNK, actualBlockShapeMNK, gmBlockC, layoutBlockC);
            }
        }
    }

private:
    // ID used for inter-core synchronization
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAicFinish;
    Catlass::Arch::Resource<ArchTag> resource;
};

}  // namespace Catcoc::DGemm::Kernel

#endif  // CATCOC_DGEMM_KERNEL_ALLGATHER_DEQUANT_MATMUL_HPP
