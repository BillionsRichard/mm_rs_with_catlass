/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATLASS_GEMM_KERNEL_MATMUL_EPILOGUE_COMM_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_EPILOGUE_COMM_HPP

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

// from kernel
#include "epilogue/block/epilogue_allreduce.hpp"
#include "epilogue/block/block_swizzle_dynamic.hpp"

namespace Catlass::Gemm::Kernel {
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
        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
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
    CATLASS_DEVICE
    MatmulEpilogueComm()
    {
        for (uint32_t i = 0; i < BufferNum; ++i) {
            flagAicFinishStore[i] = Arch::CrossCoreFlag(i);
            flagAivFinishCompute[i] = Arch::CrossCoreFlag(i);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params &params);

    template <>
    CATLASS_DEVICE
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

        // Comm need repeat
        uint32_t aicoreIndex = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkspace);

        auto blockPerComm = aicoreNum * params.pValue;
        uint32_t commCoreLoops = CeilDiv(coreLoops, blockPerComm) * blockPerComm;

        auto layoutC = layout::RowMajor{
            params.blockShape.m() * blockPerComm * BufferNum,
            params.blockShape.n(),
            params.blockShape.n()
        };

        for (uint32_t loopIdx = aicoreIndex; loopIdx < commCoreLoops; loopIdx += AscendC::GetBlockNum()) {
            uint32_t commIdx = loopIdx / blockPerComm;
            uint32_t blockLoopIdx = loopIdx / aicoreNum;
            uint32_t bufferIdx = commIdx % BufferNum;
            uint32_t pIdx = blockLoopIdx % params.pValue;

            if (pIdx == 0 && commIdx >= BufferNum) {
                Arch::CrossCoreWaitFlag(flagAivFinishCompute[bufferIdx]);
            }

            if (loopIdx < coreLoops) {
                // Compute block location
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{loopIdx % (blockPerComm * BufferNum) * L1TileShape::M, 0};
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
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[bufferIdx]);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE
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
            Arch::CrossCoreWaitFlag(flagAicFinishStore[flagIdx]);

            blockAllReduceEpilogue(blockShape, commBlockCount, actualCommBlockCount, calIdx, params.rankIdx, params.rankSize, params.pValue);

            // set aic
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[flagIdx]);
        }

    }

private:
    const static uint32_t BufferNum = 2;

    // ID used for inter-core synchronization
    Arch::CrossCoreFlag flagAicFinishStore[BufferNum];
    Arch::CrossCoreFlag flagAivFinishCompute[BufferNum];
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MATMUL_EPILOGUE_COMM_HPP
