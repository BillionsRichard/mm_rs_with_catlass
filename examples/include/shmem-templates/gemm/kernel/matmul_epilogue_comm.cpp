ifndef ACT_GEMM_KERNEL_MATMUL_EPILOGUE_COMM_HPP
#define ACT_GEMM_KERNEL_MATMUL_EPILOGUE_COMM_HPP

// from ascendc-templates
#include "act/act.hpp"
#include "act/arch/resource.hpp"
#include "act/arch/cross_core_sync.hpp"
#include "act/gemm_coord.hpp"
#include "act/matrix_coord.hpp"

// from shmem-templates
#include "shmem-templates/epilogue/block/epilogue_allreduce.hpp"
#include "shmem-templates/epilogue/block/block_swizzle_dynamic.hpp"
#include "shmem-templates/epilogue/tile/remote_copy_op.hpp"
#include "shmem-templates/arch/cross_rank_sync.hpp"

namespace Act::Gemm::Kernel {

ACT_DEVICE
MatrixCoord GetActualShape(
    const MatrixCoord &blockCount,
    const MatrixCoord &blockCoord,
    const MatrixCoord &blockShape,
    const MatrixCoord &residue
)
{
    MatrixCoord c = blockShape;

    if ((residue.row() != 0) && (blockCoord.row() == blockCount.row() - 1)) {
        c.row() = residue.row();
    }
    else if (blockCoord.row() >= blockCount.row()){
        c.row() = 0;
    }

    if ((residue.column() != 0) && (blockCoord.column() == blockCount.column() - 1)) {
        c.column() = residue.column();
    }
    else if (blockCoord.column() >= blockCount.column()){
        c.column() = 0;
    }
    return c;
}

// Template for matmul add kernel. Compute D = A * B + X
template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    bool RelaxedLenPerLoop = false
>
class MatmulEpilogueComm {
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

    using BlockEpilogue = BlockEpilogue_;
    using EpilogueParams = typename BlockEpilogue::Params;

    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GemmCoord blockShape;

        uint32_t pValue;
        uint32_t rankIdx;
        uint32_t rankSize;

        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrWorkspace;
        EpilogueParams epilogueParams;

        // Methods
        ACT_DEVICE
        Params() {}

        ACT_DEVICE
        Params(
            GemmCoord const &problemShape_,
            GemmCoord const &blockShape_,
            uint32_t pValue_, uint32_t rank_, uint32_t rankSize_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrWorkspace_, EpilogueParams &epilogueParams_
        ) : problemShape(problemShape_), blockShape(blockShape_),
            pValue(pValue_), rankIdx(rank_), rankSize(rankSize_),
            ptrA(ptrA_), layoutA(layoutA_),
            ptrB(ptrB_), layoutB(layoutB_),
            ptrWorkspace(ptrWorkspace_), epilogueParams(epilogueParams_) {}
    };

    // Methods
    ACT_DEVICE
    MatmulEpilogueComm() {}

    template <int32_t CORE_TYPE = g_coreType>
    ACT_DEVICE
    void operator()(Params &params);

    template <>
    ACT_DEVICE
    void operator()<AscendC::AIC>(Params &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);
        layout::RowMajor layoutC(params.problemShape.m(), params.problemShape.n());

        // Comm need repeat
        uint32_t aicoreIndex = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        auto commRepeat = aicoreNum * params.pValue;
        uint32_t commCoreLoops = CeilDiv(coreLoops, commRepeat) * commRepeat;

        for (uint32_t loopIdx = aicoreIndex; loopIdx < commCoreLoops; loopIdx += AscendC::GetBlockNum()) {
            uint32_t blockLoopIdx = loopIdx / aicoreNum;
            uint32_t pIdx = blockLoopIdx % params.pValue;
            if (loopIdx < coreLoops) {
                // Compute block location
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);

                // Compute block-scoped matrix multiply-add
                blockMmad(
                    gmA[gmOffsetA], params.layoutA,
                    gmB[gmOffsetB], params.layoutB,
                    gmC[gmOffsetC], layoutC,
                    actualBlockShape);
            }
            if (pIdx == params.pValue - 1) {
                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    template <>
    ACT_DEVICE
    void operator()<AscendC::AIV>(Params &params)
    {
        BlockEpilogue blockAllReduceEpilogue(resource, params.epilogueParams, params.blockShape);

        uint32_t aicoreNum = AscendC::GetBlockNum();
        auto loopNumPerComm = aicoreNum * params.pValue;
        // Split core loop to comm loop tile
        MatrixCoord coreLoops{params.epilogueParams.gemmSwizzle.GetCoreLoops(), 1};
        MatrixCoord commBlockCount{loopNumPerComm, 1};
        auto commLoops = CeilDiv(coreLoops, commBlockCount);
        auto residueCommBlockCount = coreLoops % commBlockCount;

        MatrixCoord blockShape{params.blockShape.m(), params.blockShape.n()};

        uint32_t BufferNum = 2;
        for (uint32_t calIdx = 0; calIdx < commLoops.row() * commLoops.column(); ++calIdx) {
            uint32_t flagIdx = calIdx % BufferNum;
            MatrixCoord commLoopsCoord{calIdx, 0};
            MatrixCoord actualCommBlockCount = GetActualShape(
                commLoops,
                commLoopsCoord,
                commBlockCount,
                residueCommBlockCount
            );

            // wait aic
            Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore);

            blockAllReduceEpilogue(blockShape, actualCommBlockCount, commBlockCount, calIdx, params.rankIdx, params.rankSize, params.pValue);

            // // 发送aic同步
            // arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[flagIdx]);
        }

    }

private:
    // ID used for inter-core synchronization
    static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    Arch::Resource<ArchTag> resource;

    __ubuf__ int32_t *ctrl_flags_UB = (__ubuf__ int32_t *)(131072);
};

} // namespace Act::Gemm::Kernel

#endif // ACT_GEMM_KERNEL_MATMUL_EPILOGUE_COMM_HPP
