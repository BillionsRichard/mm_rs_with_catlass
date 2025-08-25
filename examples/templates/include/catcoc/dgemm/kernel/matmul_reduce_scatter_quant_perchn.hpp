#ifndef CATCOC_DGEMM_KERNEL_MATMUL_REDUCE_SCATTER_QUANT_PERCHN_HPP
#define CATCOC_DGEMM_KERNEL_MATMUL_REDUCE_SCATTER_QUANT_PERCHN_HPP

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
            class BlockMmadScheduler_,
            class BlockEpilogueScheduler_,
            uint32_t WORKSPACE_STAGES_
    >
    class MatmulReduceScatterQuantPerchn {
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
        using ElementBias = typename BlockMmad::ElementBias;
        using ElementScale = typename BlockMmad::ElementScale;

        using BlockEpilogueReduceScatter = BlockEpilogueReduceScatter_;
        using BlockEpilogueReduceScatterParams = typename BlockEpilogueReduceScatter::Params;

        using ElementD = typename BlockEpilogueReduceScatter::ElementDst;
        using LayoutD = typename BlockEpilogueReduceScatter::LayoutDst;

        using BlockMmadScheduler = BlockMmadScheduler_;
        using BlockEpilogueScheduler = BlockEpilogueScheduler_;

        static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

        /// Parameters structure
        struct Params {
            // Data members
            GemmCoord problemShape;
            uint32_t rankIdx;
            uint32_t rankSize;
            // int32_t teamIdx;

            uint32_t commInterval;

            GM_ADDR ptrA;
            LayoutA layoutA;
            GM_ADDR ptrB;
            LayoutB layoutB;
            GM_ADDR ptrBias;
            GM_ADDR ptrScale;
            GM_ADDR ptrSymmetric;
            BlockEpilogueReduceScatterParams epilogueReduceScatter;

            GM_ADDR ptrD;
            LayoutD layoutD;

            // Methods
            CATLASS_DEVICE
            Params() {}

            CATLASS_DEVICE
            Params(
                    GemmCoord const &problemShape_,
                    uint32_t rank_, uint32_t rankSize_,
                    GM_ADDR ptrA_, LayoutA const &layoutA_,
                    GM_ADDR ptrB_, LayoutB const &layoutB_,
                    GM_ADDR ptrBias_, GM_ADDR ptrScale_,
                    GM_ADDR ptrSymmetric_,
                    BlockEpilogueReduceScatterParams const &epilogueReduceScatter_,
                    GM_ADDR ptrD_, LayoutD const &layoutD_,
                    uint32_t commInterval_
            ) : problemShape(problemShape_),
                rankIdx(rank_), rankSize(rankSize_),
                ptrA(ptrA_), layoutA(layoutA_),
                ptrB(ptrB_), layoutB(layoutB_),
                ptrBias(ptrBias_), ptrScale(ptrScale_),
                ptrSymmetric(ptrSymmetric_),
                epilogueReduceScatter(epilogueReduceScatter_),
                ptrD(ptrD_), layoutD(layoutD_),
                commInterval(commInterval_) {}
        };

        // Methods
        CATLASS_DEVICE
        MatmulReduceScatterQuantPerchn()
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
            uint32_t aicoreIndex = AscendC::GetBlockIdx();
            uint32_t aicoreNum = AscendC::GetBlockNum();
            uint32_t blockPerComm = aicoreNum * params.commInterval;
            uint32_t blockPerCommInRank = blockPerComm / params.rankSize;

            GemmCoord blockShape = L1TileShape::ToCoord();
            GemmCoord problemShapeInRank = params.problemShape / Catlass::MakeCoord<uint32_t>(params.rankSize, 1, 1);
            BlockMmadScheduler mmadScheduler(problemShapeInRank, blockShape.GetCoordMN());
            uint32_t coreLoops = mmadScheduler.GetCoreLoops() * params.rankSize;
            uint32_t commLoops = CeilDiv(coreLoops, blockPerComm);

            BlockMmad blockMmad(resource);

            // Represent the full gm
            AscendC::GlobalTensor<ElementA> gmA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            AscendC::GlobalTensor<ElementBias> gmBias;
            gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias);
            AscendC::GlobalTensor<ElementScale> gmScale;
            gmScale.SetGlobalBuffer((__gm__ ElementScale *)params.ptrScale);
            AscendC::GlobalTensor<ElementC> gmSymmetric;
            gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric));
            AscendC::GlobalTensor<ElementC> gmD;
            gmD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrD));

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

                uint32_t actualBlockPerComm = (commIdx == commLoops - 1) ?
                                              (coreLoops - blockPerComm * commIdx) : blockPerComm;
                uint32_t actualBlockPerCommInRank = actualBlockPerComm / params.rankSize;

                uint32_t commBlockOffsetInRank = commIdx * blockPerCommInRank;
                for (uint32_t blockIdxInComm = aicoreIndex; blockIdxInComm < actualBlockPerComm;
                blockIdxInComm += aicoreNum) {
                    uint32_t loopIdxInRank = commBlockOffsetInRank + blockIdxInComm % actualBlockPerCommInRank;
                    uint32_t targetRankIdx = blockIdxInComm / actualBlockPerCommInRank;
                    // Compute block location
                    GemmCoord blockCoord = mmadScheduler.GetBlockCoord(loopIdxInRank);
                    GemmCoord actualBlockShape = mmadScheduler.GetActualBlockShape(blockCoord);

                    GemmCoord offsetCoord = blockCoord * blockShape;
                    // Compute initial location in logical coordinates
                    auto rankOffsetA = problemShapeInRank.GetCoordMK() * Catlass::MakeCoord<uint32_t>(targetRankIdx, 0);
                    auto blockOffsetA = offsetCoord.GetCoordMK() + rankOffsetA;
                    auto blockOffsetB = offsetCoord.GetCoordKN();

                    MatrixCoord blockOffsetStore;
                    AscendC::GlobalTensor<ElementC> gmStore;
                    Catlass::layout::RowMajor layoutStore;
                    if (targetRankIdx == params.rankIdx) {
                        blockOffsetStore = offsetCoord.GetCoordMN();
                        gmStore = gmD;
                        layoutStore = params.layoutD;
                    }
                    else {
                        blockOffsetStore = MatrixCoord{layoutCRow(Catlass::MakeCoord<int>(stageId, blockIdxInComm, 0)), 0};
                        gmStore = gmSymmetric;
                        layoutStore = layoutC;
                    }

                    int64_t offsetA = params.layoutA.GetOffset(blockOffsetA);
                    int64_t offsetB = params.layoutB.GetOffset(blockOffsetB);
                    int64_t offsetStore = layoutStore.GetOffset(blockOffsetStore);
                    int64_t gmOffsetBias = blockCoord.n() * L1TileShape::N;
                    int64_t gmOffsetScale = blockCoord.n() * L1TileShape::N;

                    // Compute block-scoped matrix multiply-add
                    blockMmad(
                        gmA[offsetA], params.layoutA,
                        gmB[offsetB], params.layoutB,
                        gmStore[offsetStore], layoutStore,
                        gmBias[gmOffsetBias], gmScale[gmOffsetScale],
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
            uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            uint32_t aicoreNum = AscendC::GetBlockNum();
            uint32_t aivIndex = AscendC::GetSubBlockIdx();
            uint32_t blockPerComm = aicoreNum * params.commInterval;
            uint32_t blockPerCommInRank = blockPerComm / params.rankSize;

            MatrixCoord blockShapeMN = L1TileShape::ToCoordMN();
            GemmCoord problemShapeInRank = params.problemShape / Catlass::MakeCoord<uint32_t>(params.rankSize, 1, 1);
            BlockMmadScheduler mmadScheduler(problemShapeInRank, blockShapeMN);
            uint32_t coreLoops = mmadScheduler.GetCoreLoops() * params.rankSize;
            auto commLoops = CeilDiv(coreLoops, blockPerComm);

            BlockEpilogueReduceScatter epilogueReduceScatter(resource, params.epilogueReduceScatter);

            AscendC::GlobalTensor<ElementC> gmSymmetric;
            gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric));
            auto layoutSymmetric = Catlass::layout::RowMajor{
                    WORKSPACE_STAGES * blockPerComm * L1TileShape::M, L1TileShape::N,
                    L1TileShape::N
            };

            AscendC::GlobalTensor<ElementD> gmD;
            gmD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD *>(params.ptrD));

            MatrixCoord commBlockShape = params.epilogueReduceScatter.BlockShape();
            MatrixCoord commCoreSplit = params.epilogueReduceScatter.CoreSplit();
            BlockEpilogueScheduler commScheduler(commBlockShape, commCoreSplit);
            for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
                uint32_t stageIdx = commIdx % WORKSPACE_STAGES;
                uint32_t actualBlockInComm = Min(blockPerComm, coreLoops - commIdx * blockPerComm);
                auto actualCommShape =
                        DistMatrixCoord{actualBlockInComm * blockShapeMN.row() / params.rankSize, blockShapeMN.column(), params.rankSize};
                MatrixCoord loopsInRank = CeilDiv(MatrixCoord(actualCommShape.GetCoordInRank()), commBlockShape);

                commScheduler.UpdateProblem(actualCommShape, loopsInRank);
                uint32_t commAicoreNum = commScheduler.GetRealCore();
                uint32_t commCoreLoops = commScheduler.GetCoreLoop();

                MatrixCoord stageOffset = MatrixCoord{stageIdx * blockPerComm, 0} * blockShapeMN;
                uint32_t mmadStartLoopIdxInComm = commIdx * blockPerCommInRank;

                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);

                // Local matmul is completed, waiting until tasks on all devices are complete.
                shmemx_barrier_all_vec();

                AscendC::SetAtomicAdd<ElementD>();
                AscendC::PipeBarrier<PIPE_ALL>();
                epilogueReduceScatter.InitBlockLoop();
                if (aivIndex == 0 && aicoreIndex < commAicoreNum) {
                    for (uint32_t commLoopIdx = aicoreIndex; commLoopIdx < commCoreLoops; commLoopIdx += commAicoreNum) {
                        DistMatrixCoord commBlockCoord = commScheduler.GetBlockCoord(commLoopIdx);
                        MatrixCoord blockOffset = commScheduler.GetBlockOffset(
                            DistMatrixCoord{commBlockCoord.GetCoordInRank(), params.rankIdx});
                        MatrixCoord blockOffsetInRank = commScheduler.GetBlockOffsetInRank(commBlockCoord.GetCoordInRank());
                        MatrixCoord actualCommBlockShape = commScheduler.GetActualBlockShapeByOffset(blockOffsetInRank);

                        uint32_t remoteRankIdx = commBlockCoord.rank();
                        if (remoteRankIdx == params.rankIdx) {
                            continue;
                        }

                        uint32_t mmadLoopIdx = mmadStartLoopIdxInComm + blockOffsetInRank.row() / blockShapeMN.row();
                        GemmCoord mmadBlockCoordMNK = mmadScheduler.GetBlockCoord(mmadLoopIdx);
                        MatrixCoord mmadBlockCoord = mmadBlockCoordMNK.GetCoordMN();
                        MatrixCoord actualMmadBlockShape =
                                mmadScheduler.GetActualBlockShape(mmadBlockCoordMNK).GetCoordMN();

                        MatrixCoord offsetInMmadBlock = blockOffsetInRank % blockShapeMN;
                        MatrixCoord residueInMmadBlock = actualMmadBlockShape -
                                                         Min<uint32_t, 2>(actualMmadBlockShape, offsetInMmadBlock);
                        actualCommBlockShape = Min<uint32_t, 2>(actualCommBlockShape, residueInMmadBlock);

                        auto offsetSrc = stageOffset + blockOffset;
                        MatrixCoord mmadBlockOffset = mmadBlockCoord * blockShapeMN;
                        auto offsetDst = mmadBlockOffset + offsetInMmadBlock;

                        auto gmBlockSrc = gmSymmetric[layoutSymmetric.GetOffset(offsetSrc)];
                        auto layoutBlockSrc = layoutSymmetric.GetTileLayout(actualCommBlockShape);

                        auto gmBlockDst = gmD[params.layoutD.GetOffset(offsetDst)];
                        auto layoutBlockDst = params.layoutD.GetTileLayout(actualCommBlockShape);

                        epilogueReduceScatter(
                                gmBlockSrc, layoutBlockSrc,
                                gmBlockDst, layoutBlockDst,
                                actualCommBlockShape, remoteRankIdx % params.rankSize
                        );
                    }
                }
                epilogueReduceScatter.FinalizeBlockLoop();
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                AscendC::SetAtomicNone();
                AscendC::PipeBarrier<PIPE_ALL>();

                // ReduceScatter is completed, waiting until tasks on all devices are complete.
                shmemx_barrier_all_vec();

                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageIdx]);
            }
        }

    private:
        // ID used for inter-core synchronization
        Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
        Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
        Catlass::Arch::Resource<ArchTag> resource;
    };

} // namespace Catcoc::Gemm::Kernel

#endif // CATCOC_DGEMM_KERNEL_MATMUL_REDUCE_SCATTER_QUANT_PERCHN_HPP
