#ifndef CATCOC_DGEMM_KERNEL_QUANT_MATMUL_REDUCE_SCATTER_HPP
#define CATCOC_DGEMM_KERNEL_QUANT_MATMUL_REDUCE_SCATTER_HPP

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
    class BlockEpilogueBias_,
    class BlockEpilogueDequant_,
    class BlockScheduler_,
    class BlockEpilogueScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class QuantMatmulReduceScatter {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC; // int32_t accumulator
    using LayoutC = typename BlockMmad::LayoutC;

    using ReduceScatter = BlockEpilogueReduceScatter_;
    using ReduceScatterParams = typename ReduceScatter::Params;
    using BiasAdd = BlockEpilogueBias_;
    using BiasParams = typename BiasAdd::Params;
    using Dequant = BlockEpilogueDequant_;
    using DequantParams = typename Dequant::Params;

    using ElementD = bfloat16_t; // Final output type
    using LayoutD = Catlass::layout::RowMajor;

    using BlockScheduler = BlockScheduler_;
    using CommScheduler = BlockEpilogueScheduler_;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;

        uint32_t rankIdx;
        uint32_t rankSize;

        GM_ADDR ptrA; // int8
        LayoutA layoutA;
        GM_ADDR ptrB; // int8
        LayoutB layoutB;
        GM_ADDR ptrSymmetric;
        ReduceScatterParams reduceScatterParams;
        BiasParams biasParams;
        DequantParams dequantParams;

        GM_ADDR ptrC_accum; // int32
        LayoutC layoutC_accum;
        
        GM_ADDR ptrD_out; // bfloat16
        LayoutD layoutD_out;

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
            ReduceScatterParams const &reduceScatterParams_,
            BiasParams const &biasParams_,
            DequantParams const &dequantParams_,
            GM_ADDR ptrC_accum_, LayoutC const &layoutC_accum_,
            GM_ADDR ptrD_out_, LayoutD const &layoutD_out_,
            uint32_t commInterval_
        ) : problemShape(problemShape_),
            rankIdx(rank_), rankSize(rankSize_),
            ptrA(ptrA_), layoutA(layoutA_),
            ptrB(ptrB_), layoutB(layoutB_),
            ptrSymmetric(ptrSymmetric_),
            reduceScatterParams(reduceScatterParams_),
            biasParams(biasParams_),
            dequantParams(dequantParams_),
            ptrC_accum(ptrC_accum_), layoutC_accum(layoutC_accum_),
            ptrD_out(ptrD_out_), layoutD_out(layoutD_out_),
            commInterval(commInterval_) {}
    };

    // Methods
    CATLASS_DEVICE
    QuantMatmulReduceScatter()
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
        BlockScheduler matmulBlockScheduler(problemShapeInRank, blockShape.GetCoordMN());
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops() * params.rankSize;
        uint32_t commLoops = CeilDiv(coreLoops, blockPerComm);

        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
        AscendC::GlobalTensor<ElementC> gmC_workspace;
        gmC_workspace.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementC> gmC_accum;
        gmC_accum.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC_accum));

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
            for (
                uint32_t blockIdxInComm = aicoreIndex;
                blockIdxInComm < actualBlockPerComm;
                blockIdxInComm += aicoreNum
            ) {
                uint32_t loopIdxInRank = commBlockOffsetInRank + blockIdxInComm % actualBlockPerCommInRank;
                uint32_t targetRankIdx = blockIdxInComm / actualBlockPerCommInRank;
                // Compute block location
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdxInRank);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

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
                    gmStore = gmC_accum;
                    layoutStore = params.layoutC_accum;
                }
                else {
                    blockOffsetStore = MatrixCoord{layoutCRow(Catlass::MakeCoord<int>(stageId, blockIdxInComm, 0)), 0};
                    gmStore = gmC_workspace;
                    layoutStore = layoutC;
                }
                
                int64_t offsetA = params.layoutA.GetOffset(blockOffsetA);
                int64_t offsetB = params.layoutB.GetOffset(blockOffsetB);
                int64_t offsetStore = layoutStore.GetOffset(blockOffsetStore);
                
                // Compute block-scoped matrix multiply-add
                blockMmad(
                    gmA[offsetA], params.layoutA,
                    gmB[offsetB], params.layoutB,
                    gmStore[offsetStore], layoutStore,
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
        BlockScheduler matmulBlockScheduler(problemShapeInRank, blockShapeMN);
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops() * params.rankSize;
        auto commLoops = CeilDiv(coreLoops, blockPerComm);

        ReduceScatter reduceScatter(resource, params.reduceScatterParams);

        AscendC::GlobalTensor<ElementC> gmC_workspace;
        gmC_workspace.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric));

        AscendC::GlobalTensor<ElementC> gmC_accum;
        gmC_accum.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC_accum));

        MatrixCoord commBlockShape = params.reduceScatterParams.BlockShape();
        MatrixCoord commCoreSplit = params.reduceScatterParams.CoreSplit();
        MatrixCoord commShape = MatrixCoord{blockPerComm, 1} * blockShapeMN;
        MatrixCoord dataLoopsMx = CeilDiv(commShape, commBlockShape);
        uint32_t dLoopsInRank = CeilDiv(dataLoopsMx.row() * dataLoopsMx.column(), params.rankSize);
        CommScheduler commScheduler(params.rankIdx, params.rankSize, commCoreSplit, 
                                    commShape, commBlockShape, dLoopsInRank);
        MatrixCoord actualCommShapeInRank = commShape / Catlass::MakeCoord<uint32_t>(params.rankSize, 1);

        auto layoutCommLogicShape = Catlass::MakeCoord<int>(1, dLoopsInRank, commBlockShape.row());
        auto layoutComm = layout::AffineRankN<3>::Packed(layoutCommLogicShape);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;
            if (commIdx == commLoops - 1) {
                uint32_t actualBlockInComm = coreLoops - commIdx * blockPerComm;
                commShape = MatrixCoord{actualBlockInComm, 1} * blockShapeMN;
                dataLoopsMx = CeilDiv(commShape, commBlockShape);
                dLoopsInRank = CeilDiv(dataLoopsMx.row() * dataLoopsMx.column(), params.rankSize);
                commScheduler.Update(commShape, commBlockShape, dLoopsInRank);
                layoutCommLogicShape = Catlass::MakeCoord<int>(1, dLoopsInRank, commBlockShape.row());
                layoutComm = layout::AffineRankN<3>::Packed(layoutCommLogicShape);
                actualCommShapeInRank = commShape / Catlass::MakeCoord<uint32_t>(params.rankSize, 1);
            }
            auto commAicoreNum = commScheduler.GetRealCore();
            auto commCoreLoops = commScheduler.GetCoreLoop();

            MatrixCoord stageOffset = MatrixCoord{stageId * blockPerComm, 0} * blockShapeMN;
            MatrixCoord commOffsetInRank = MatrixCoord{commIdx * blockPerCommInRank, 0} * blockShapeMN;

            // wait aic
            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageId]);
            shmemx_barrier_all_vec();

            AscendC::SetAtomicAdd<ElementC>(); // AtomicAdd on int32
            AscendC::PipeBarrier<PIPE_ALL>();
            reduceScatter.AllocEventID();
            if (aivIndex == 0 && aicoreIndex < commAicoreNum) {
                for (uint32_t commLoopIdx = aicoreIndex; commLoopIdx < commCoreLoops; commLoopIdx += commAicoreNum) {
                    MatrixCoord commBlockCoord = commScheduler.GetBlockIdx(commLoopIdx);
                    MatrixCoord blockOffset = commScheduler.template GetBlockOffset<ReduceScatter::RemoteCopyMode,
                        ReduceScatter::RemoteCopyDirect>(commBlockCoord, layoutComm);
                    MatrixCoord actualCommBlockShape = commScheduler.template GetActualBlockShape<
                        ReduceScatter::RemoteCopyMode, ReduceScatter::RemoteCopyDirect>(commBlockCoord, layoutComm);
                    MatrixCoord blockOffsetInRank = blockOffset % actualCommShapeInRank;

                    uint32_t remoteRankIdx = commBlockCoord.column();
                    if (remoteRankIdx == params.rankIdx) {
                        continue;
                    }

                    auto offsetIn = stageOffset + blockOffset;
                    auto offsetOut = commOffsetInRank + blockOffsetInRank;

                    auto globalLoopIdx = offsetOut.row() / blockShapeMN.row();

                    reduceScatter(blockShapeMN, offsetOut, offsetIn, actualCommBlockShape,
                        gmC_accum, params.layoutC_accum, globalLoopIdx, remoteRankIdx % params.rankSize);
                }
            }
            reduceScatter.ReleaseEventID();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::SetAtomicNone();
            AscendC::PipeBarrier<PIPE_ALL>();

            // ReduceScatter is completed, waiting until tasks on all devices are complete.
            shmemx_barrier_all_vec();

            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId]);
        }

        uint32_t M_per_rank = params.problemShape.m() / params.rankSize;
        uint32_t N = params.problemShape.n();
        GemmCoord problemShapeEpilogue{M_per_rank, N, 1};
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

        // BiasAdd Step
        if (params.biasParams.ptr_bias != 0) {
            {
                BiasAdd biasAddEpilogue(resource, params.biasParams);
                biasAddEpilogue(
                    problemShapeEpilogue,
                    GemmCoord{0, 0, 0},
                    problemShapeEpilogue,
                    gmC_accum,
                    params.layoutC_accum
                );
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }

        // Final Dequantization Step, using the epilogue
        Dequant dequantEpilogue(resource, params.dequantParams);

        // Use the epilogue's own tile scheduler to iterate over the output matrix
        auto cord = Dequant::TileShape::ToCoord();
        typename Dequant::EpilogueTileSwizzle tileScheduler(problemShapeEpilogue.GetCoordMN(), cord);
        uint32_t tileLoops = tileScheduler.GetLoops();

        for(uint32_t i = coreIdx; i < tileLoops; i += coreNum) {
            auto tileCoord = tileScheduler.GetTileCoord(i);
            auto actualTileShape = tileScheduler.GetActualTileShape(tileCoord);
            auto acc_offset = tileCoord * cord;
            // The epilogue call must be adapted to its own scheduling logic
            dequantEpilogue(
                GemmCoord(cord[0], cord[1], 1),
                GemmCoord(tileCoord.row(), tileCoord.column(), 0), 
                GemmCoord(actualTileShape.row(), actualTileShape.column(), 1), 
                gmC_accum[params.layoutC_accum.GetOffset(acc_offset)], 
                params.layoutC_accum.GetTileLayout(actualTileShape)
            );
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    // ID used for inter-core synchronization
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace Catcoc::Gemm::Kernel

#endif // CATCOC_DGEMM_KERNEL_QUANT_MATMUL_REDUCE_SCATTER_HPP
