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

// from shmem-device
#include "shmem_api.h"

using namespace Act;

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

namespace Act::Epilogue::Block {
template <
    class... Args
>
class EpilogueAllReduce {

};

template <
    class BlockScheduler_,
    class CommBlockSwizzle_
>
class EpilogueAllReduce <
    BlockScheduler_,
    CommBlockSwizzle_
> {
public:
    using BlockScheduler = BlockScheduler_;
    using CommBlockSwizzle = CommBlockSwizzle_;

    // Type aliases
    using ArchTag = Arch::AtlasA2;

    using ElementC = half;
    using ElementAttachedSource = half;
    using ElementAttachedOutput = half;

    using LayoutStore = layout::RowMajor;

    using LayoutWorkspace = LayoutStore;

    using ElementDestination = ElementAttachedOutput;

    using LayoutDestination = LayoutStore;

    using ScheduleTypeOp1 = typename Gemm::Block::ReduceScatterSchedule;
    using ScheduleTypeOp2 = typename Gemm::Block::AllGatherSchedule;

    // Epilogue params definition
    struct Params {
        AscendC::GlobalTensor<ElementDestination> destination;
        LayoutDestination layoutDestination;
        int64_t strideDestination;
        __gm__ ElementAttachedSource *symmetricPtr;
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
            __gm__ ElementAttachedSource *symmetricPtr,
            MatrixCoord blockShape,
            MatrixCoord processShape,
            BlockScheduler gemmSwizzle,
            CommBlockSwizzle commSwizzle
        ) :
            destination(destination),
            layoutDestination(layoutDestination),
            strideDestination(strideDestination),
            symmetricPtr(symmetricPtr),
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

        AscendC::GlobalTensor<ElementC> peerMem;
        peerMem.SetGlobalBuffer(params.symmetricPtr);

        // 卡内matmul结果准备就绪软同步
        shmem_barrier_all();

        AscendC::SetAtomicAdd<ElementC>();
        AscendC::PipeBarrier<PIPE_ALL>();

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

                auto offsetIn = blockOffset + rankBlockOffset;
                auto offsetOut = blockOffset + rankBlockOffset;

                auto residueProcessShape = actualCommSubBlockShape % params.processShape;
                auto processCount = CeilDiv(actualCommSubBlockShape, params.processShape);
                uint32_t processLoop = processCount.row() * processCount.column();

                // [ReduceScatter] 1. Alloc TmpUB
                AscendC::LocalTensor<half> inputBuffer = resource.ubBuf.template GetBufferByByte<ElementC>(0);

                // [ReduceScatter] 2. Pre Interface Sync
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

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

                    uint32_t copySize = actualProcessShape.row() * actualProcessShape.column();

                    int64_t inputElemOffset = layoutPeerMemStore.GetOffset(inputOffset);
                    int64_t outputElemOffset = layoutPeerMemStore.GetOffset(outputOffset);

                    // [ReduceScatter] 2. Pre Interface Sync
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

                    // [ReduceScatter] 3. Start shmem_mte_get_mem_nbi
                    shmem_mte_get_mem_nbi(peerMem[outputElemOffset], peerMem[inputElemOffset], inputBuffer, copySize, mRankIdx % rankSize, EVENT_ID0);
                    
                    // [ReduceScatter] 4. Post Interface Sync
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
                // [ReduceScatter] 4. Post Interface Sync
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0); // Scalar等MTE3
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetAtomicNone();
        AscendC::PipeBarrier<PIPE_ALL>();

        // 第一部分通信完成软同步
        shmem_barrier_all();

        if (aivIndex == 0 && aicoreIndex < realAicoreNum) {
            for (uint32_t idx = aicoreIndex; idx < commCoreLoops; idx += realAicoreNum) {
                MatrixCoord idxTile = params.commSwizzle.GetBlockIdx(idx);
                MatrixCoord actualCommSubBlockShape = params.commSwizzle.template GetBlockSize<ScheduleTypeOp2>(idxTile);
                MatrixCoord rankBlockOffset = params.commSwizzle.template GetRankOffset<ScheduleTypeOp2>(idxTile);
                MatrixCoord subBlockOffset = params.commSwizzle.GetBlockOffset(idxTile);

                uint32_t mRankIdx = idxTile.column();

                auto offsetIn = blockOffset + rankBlockOffset;
                auto offsetOut = outputBlockOffset + rankBlockOffset;

                auto residueProcessShape = actualCommSubBlockShape % params.processShape;
                auto processCount = CeilDiv(actualCommSubBlockShape, params.processShape);

                uint32_t processLoop = processCount.row() * processCount.column();

                // [AllGather] 1. Alloc TmpUB
                AscendC::LocalTensor<half> inputBuffer = resource.ubBuf.template GetBufferByByte<ElementC>(0);

                // [AllGather] 2. Pre Interface Sync
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

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

                    uint32_t copySize = actualProcessShape.row() * actualProcessShape.column();

                    int64_t inputElemOffset = layoutPeerMemStore.GetOffset(inputOffset);
                    int64_t outputElemOffset = layoutPeerMemStore.GetOffset(outputOffset);

                    // [AllGather] 2. Pre Interface Sync
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

                    // [AllGather] 3. Start shmem_mte_get_mem_nbi
                    shmem_mte_get_mem_nbi(params.destination[outputElemOffset], peerMem[inputElemOffset], inputBuffer, copySize, mRankIdx % rankSize, EVENT_ID0);

                    // [AllGather] 4. Post Interface Sync
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
                // [AllGather] 4. Post Interface Sync
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }

        // 第二部分通信完成软同步
        shmem_barrier_all();
    }

private:
    MatrixCoord gemmBlockShape;
    Arch::Resource<ArchTag> &resource;
    Params params;
};

}  // namespace Act::Epilogue::Block

#endif  // _EPILOGUE_ALLREDUCE_HPP