#ifndef _EPILOGUE_ALLREDUCE_HPP
#define _EPILOGUE_ALLREDUCE_HPP
// from ascendc-templates
#include "act/act.hpp"
#include "act/arch/resource.hpp"
#include "act/epilogue/dispatch_policy.hpp"
#include "act/gemm_coord.hpp"
#include "act/matrix_coord.hpp"
#include "act/layout/layout.hpp"

// from shmem-templates
#include "shmem-templates/epilogue/block/block_swizzle_dynamic.hpp"
#include "shmem-templates/epilogue/tile/remote_copy_op.hpp"
#include "shmem-templates/arch/cross_rank_sync.hpp"

namespace Act::Epilogue::Block {

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

template <
    class... Args
>
class EpilogueAllReduce {

};

template <
    class BlockScheduler_,
    class CommBlockSwizzle_,
    class ComputeAttachedOp1_,
    class ComputeAttachedOp2_
>
class EpilogueAllReduce <
    BlockScheduler_,
    CommBlockSwizzle_,
    ComputeAttachedOp1_,
    ComputeAttachedOp2_
> {
public:
    using BlockScheduler = BlockScheduler_;
    using CommBlockSwizzle = CommBlockSwizzle_;
    using ComputeAttachedOp1 = ComputeAttachedOp1_;
    using ComputeAttachedOp2 = ComputeAttachedOp2_;

    // Type aliases
    using ArchTag = Arch::AtlasA2;

    using ElementC = typename ComputeAttachedOp1::ElementCompute;
    using ElementStore = typename ComputeAttachedOp1::ElementCompute;
    using ElementAttachedSource = typename ComputeAttachedOp1::ElementCompute;
    using ElementAttachedOutput = typename ComputeAttachedOp1::ElementCompute;

    using LayoutStore = layout::RowMajor;
    using LayoutAttachedSource = LayoutStore;

    using ElementWorkspace = ElementStore;
    using LayoutWorkspace = LayoutStore;

    using ElementDestination = ElementAttachedOutput;

    using LayoutDestination = LayoutStore;

    using ScheduleTypeOp1 = typename ComputeAttachedOp1::ScheduleType;
    using ScheduleTypeOp2 = typename ComputeAttachedOp2::ScheduleType;

    // Epilogue params definition
    struct Params {
        AscendC::GlobalTensor<ElementDestination> destination;
        LayoutDestination layoutDestination;
        int64_t strideDestination;
        __gm__ ElementAttachedSource **buff;
        BlockScheduler gemmSwizzle;
        CommBlockSwizzle commSwizzle;
        MatrixCoord blockShape;
        MatrixCoord processShape;

        ACT_DEVICE
        Params() = default;

        ACT_DEVICE
        Params(
            AscendC::GlobalTensor<ElementDestination> destination,
            const LayoutDestination &layoutDestination,
            int64_t strideDestination,
            __gm__ ElementAttachedSource **buff,
            MatrixCoord blockShape,
            MatrixCoord processShape,
            BlockScheduler gemmSwizzle,
            CommBlockSwizzle commSwizzle
        ) :
            destination(destination),
            layoutDestination(layoutDestination),
            strideDestination(strideDestination),
            buff(buff),
            blockShape(blockShape),
            processShape(processShape),
            gemmSwizzle(gemmSwizzle),
            commSwizzle(commSwizzle)
        {}
    };

    ACT_DEVICE
    EpilogueAllReduce(
        Arch::Resource<ArchTag> &resource,
        Params const &params,
        const GemmCoord &blockGemmShape) :
        resource(resource),
        params(params),
        gemmBlockShape(blockGemmShape.m(), blockGemmShape.n()) {}

    ACT_DEVICE
    ~EpilogueAllReduce() {}

    ACT_DEVICE
    void operator() (
        MatrixCoord const &blockShape,
        MatrixCoord const &commBlockCount,
        MatrixCoord const &actualCommBlockCount,
        uint32_t calIdx,
        uint32_t rankIdx,
        uint32_t rankSize,
        uint32_t pValue
    )
    {
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        int32_t aivIndex = AscendC::GetSubBlockIdx();
        auto loopNumPerComm = aicoreNum * pValue;

        uint32_t BufferNum = 2;
        auto layoutPeerMemStore = layout::RowMajor(
            blockShape.row() * loopNumPerComm * BufferNum,
            blockShape.column(),
            blockShape.column()
        );

        //数据打平，重新按照通信进行切分block
        uint32_t flagIdx = calIdx % BufferNum;
        MatrixCoord actualCommBlockShape = blockShape * actualCommBlockCount;
        MatrixCoord outputBlockOffset = blockShape * MatrixCoord{calIdx * loopNumPerComm, 0};

        MatrixCoord commCoordPeerMem{flagIdx, 0};
        MatrixCoord blockOffset = commCoordPeerMem * commBlockCount * blockShape;

        params.commSwizzle.template
            SetProblemSize<
                ScheduleTypeOp1, true>(actualCommBlockShape);
        auto realAicoreNum = params.commSwizzle.GetRealCore();
        auto commCoreLoops = params.commSwizzle.GetCoreLoop();

        AscendC::GlobalTensor<ElementC> peerMemOut;
        peerMemOut.SetGlobalBuffer(params.buff[rankIdx]);

        // 卡内matmul结果准备就绪软同步
        Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        Arch::CrossRankSync(Arch::FLAG_ZERO_IDX, calIdx + 1, rankIdx, rankSize, ctrl_flags_UB, (__gm__ int32_t **)params.buff);
        Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

        SetAtomicAdd<ElementC>();
        PipeBarrier<PIPE_ALL>();

        if (aivIndex == 0 && aicoreIndex < realAicoreNum) {                 // 只启用一个aiv核
            for (uint32_t idx = aicoreIndex; idx < commCoreLoops; idx += realAicoreNum) {
                MatrixCoord idxTile = params.commSwizzle.GetBlockIdx(idx);
                MatrixCoord actualCommSubBlockShape = params.commSwizzle.template GetBlockSize<ScheduleTypeOp1>(idxTile);
                MatrixCoord rankBlockOffset = params.commSwizzle.template GetRankOffset<ScheduleTypeOp1>(idxTile);
                MatrixCoord subBlockOffset = params.commSwizzle.GetBlockOffset(idxTile);
                uint32_t mRankIdx = idxTile.column();
                if (mRankIdx == rankIdx) {
                    continue;
                }
                AscendC::GlobalTensor<ElementC> peerMemIn;
                peerMemIn.SetGlobalBuffer(params.buff[mRankIdx]);
                auto offsetIn = blockOffset + rankBlockOffset;
                auto offsetOut = blockOffset + rankBlockOffset;

                auto residueProcessShape = actualCommSubBlockShape % params.processShape;
                auto processCount = CeilDiv(actualCommSubBlockShape, params.processShape);
                uint32_t processLoop = processCount.row() * processCount.column();

                AscendC::LocalTensor<half> inputBuffer = resource.ubBuf.template GetBufferByByte<ElementCompute>(32);

                // tileEpilogueCopyOp1.AllocBuffer(resource);
                // tileEpilogueCopyOp1.AllocEventID();
                for (uint32_t processIndex = 0; processIndex < processLoop; ++processIndex) {
                    MatrixCoord processCoord{processIndex / processCount.column(), processIndex % processCount.column()};
                    auto actualProcessShape = GetActualShape(
                        processCount,
                        processCoord,
                        params.processShape,
                        residueProcessShape
                    );

                    auto processOffset = processCoord * params.processShape;

                    auto inputOffset = offsetIn + subBlockOffset + processOffset;
                    auto outputOffset = offsetOut + subBlockOffset + processOffset;

                    uint32_t ubSize = 16 * 1024;
                    uint32_t copySize = actualProcessShape.row() * actualProcessShape.column();

                    ShmemMTEPutMem(peerMemOut + outputOffset, peerMemIn + inputOffset, inputBuffer, ubSize, copySize, i % rankSize, EVENT_ID0);

                    // tileEpilogueCopyOp1(
                    //     peerMemOut[layoutPeerMemStore.GetOffset(outputOffset)],
                    //     peerMemIn[layoutPeerMemStore.GetOffset(inputOffset)],
                    //     layoutPeerMemStore.GetTileLayout(actualProcessShape),
                    //     layoutPeerMemStore.GetTileLayout(actualProcessShape));
                }
                // tileEpilogueCopyOp1.ReleaseEventID();
            }
        }

        SetFlag<HardEvent::MTE3_S>(EVENT_ID0); // Scalar等MTE3
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        SetAtomicNone();
        PipeBarrier<PIPE_ALL>();

        // 第一部分通信完成软同步
        Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        Arch::CrossRankSync(Arch::FLAG_ONE_IDX, calIdx + 1, rankIdx, rankSize, ctrl_flags_UB, (__gm__ int32_t **)params.buff);
        Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

        if (aivIndex == 0 && aicoreIndex < realAicoreNum) {
            for (uint32_t idx = aicoreIndex; idx < commCoreLoops; idx += realAicoreNum) {
                MatrixCoord idxTile = params.commSwizzle.GetBlockIdx(idx);
                MatrixCoord actualCommSubBlockShape = params.commSwizzle.template GetBlockSize<ScheduleTypeOp2>(idxTile);
                MatrixCoord rankBlockOffset = params.commSwizzle.template GetRankOffset<ScheduleTypeOp2>(idxTile);
                MatrixCoord subBlockOffset = params.commSwizzle.GetBlockOffset(idxTile);

                uint32_t mRankIdx = idxTile.column();

                AscendC::GlobalTensor<ElementC> peerMemIn;
                peerMemIn.SetGlobalBuffer(params.buff[mRankIdx]);
                auto offsetIn = blockOffset + rankBlockOffset;
                auto offsetOut = outputBlockOffset + rankBlockOffset;

                auto residueProcessShape = actualCommSubBlockShape % params.processShape;
                auto processCount = CeilDiv(actualCommSubBlockShape, params.processShape);

               uint32_t processLoop = processCount.row() * processCount.column();

                tileEpilogueCopyOp2.AllocBuffer(resource);
                tileEpilogueCopyOp2.AllocEventID();
                for (uint32_t processIndex = 0; processIndex < processLoop; ++processIndex) {
                    MatrixCoord processCoord{processIndex / processCount.column(), processIndex % processCount.column()};
                    auto actualProcessShape = GetActualShape(
                        processCount,
                        processCoord,
                        params.processShape,
                        residueProcessShape
                    );

                    auto processOffset = processCoord * params.processShape;

                    uint32_t residueM = actualProcessShape.row();

                    auto inputOffset = offsetIn + subBlockOffset + processOffset;
                    auto outputOffset = offsetOut + subBlockOffset + processOffset;

                    ShmemMTEPutMem(peerMemOut, peerMemIn, inputBuffer, ubSize, copySize, i % rankSize, EVENT_ID0);

                    // tileEpilogueCopyOp2(
                    //     params.destination[layoutPeerMemStore.GetOffset(outputOffset)],
                    //     peerMemIn[layoutPeerMemStore.GetOffset(inputOffset)],
                    //     layoutPeerMemStore.GetTileLayout(actualProcessShape),
                    //     layoutPeerMemStore.GetTileLayout(actualProcessShape));
                }
                tileEpilogueCopyOp2.ReleaseEventID();
            }
        }

        // 第二部分通信完成软同步
        Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        Arch::CrossRankSync(Arch::FLAG_TWO_IDX, calIdx + 1, rankIdx, rankSize, ctrl_flags_UB, (__gm__ int32_t **)params.buff);
        Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
    }

private:
    MatrixCoord gemmBlockShape;
    Arch::Resource<ArchTag> &resource;
    Params params;

    __ubuf__ int32_t *ctrl_flags_UB = (__ubuf__ int32_t *)(131072);

    ComputeAttachedOp1 tileEpilogueCopyOp1;
    ComputeAttachedOp2 tileEpilogueCopyOp2;
};

}  // namespace Act::Epilogue::Block

#endif  // _EPILOGUE_ALLREDUCE_HPP