/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Block {

struct ReduceScatterSchedule {};
struct AllGatherSchedule {};
struct AllReduceSchedule {};

struct CommBlockSwizzleDynamic {

    ///
    /// Data members
    ///

    uint32_t swizzleOffset, swizzleDirection;
    MatrixCoord problemSize;
    MatrixCoord problemSizePerRank;
    uint32_t mLoops, nLoops;
    MatrixCoord blockShape;
    uint32_t rankSize, rankIdx;
    uint32_t nStride;
    uint32_t commDataSplit;
    uint32_t commNpuSplit;

    ///
    /// Methods
    ///

    CATLASS_DEVICE
    CommBlockSwizzleDynamic() {}

    CATLASS_DEVICE
    CommBlockSwizzleDynamic(MatrixCoord blockShape_, uint32_t rankIdx_, uint32_t rankSize_, uint32_t swizzleDirection_ = 0,
        uint32_t commDataSplit_ = 1, uint32_t commNpuSplit_ = 1)
        : blockShape(blockShape_), rankIdx(rankIdx_), rankSize(rankSize_), swizzleDirection(swizzleDirection_),
          commDataSplit(commDataSplit_), commNpuSplit(commNpuSplit_)
    {
        if (swizzleDirection == 1) {
            swizzleOffset = commNpuSplit;
        }
        else {
            swizzleOffset = commDataSplit;
        }
        nLoops = rankSize;
        nStride = rankSize / commNpuSplit;
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoop() const
    {
        return mLoops * nLoops;
    }

    template <typename CommOp, bool Align=false> CATLASS_DEVICE
    void SetProblemSize(MatrixCoord problemSize_)
    {
        problemSize = problemSize_;
        MatrixCoord commRankCount{rankSize, 1};
        if constexpr (std::is_same<CommOp, AllReduceSchedule>::value) {
            problemSizePerRank = problemSize;
        }
        else {
            problemSizePerRank = CeilDiv(problemSize, commRankCount);
            if constexpr (Align) {
                problemSizePerRank =
                    {RoundUp<uint32_t>(problemSizePerRank.row(), blockShape.row()), problemSizePerRank.column()};
            }
        }
        mLoops = CeilDiv(problemSizePerRank.row(), blockShape.row());
    }

    CATLASS_DEVICE
    uint32_t GetRealCore() const
    {
        return commDataSplit * commNpuSplit;
    }

    CATLASS_DEVICE
    MatrixCoord GetBlockIdx(uint32_t taskIdx) const {
        uint32_t innerIdx = taskIdx % (mLoops * nLoops);
        if (swizzleDirection == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(mLoops, swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * nLoops);
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * nLoops);

            uint32_t nRow = swizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = mLoops - swizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;
            nIdx = (nIdx + mIdx) % nLoops;

            return MatrixCoord{mIdx, nIdx};
        } else if (swizzleDirection == 1) { // Nz
            uint32_t tileBlockLoop = CeilDiv(nLoops, swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * mLoops);
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * mLoops);

            uint32_t nCol = swizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = nLoops - swizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nCol;

            nIdx = (nIdx * nStride) % nLoops + (nIdx * nStride) / nLoops;
            nIdx = (nIdx + mIdx) % nLoops;

            return MatrixCoord{mIdx, nIdx};
        }
        return MatrixCoord{};
    }

    CATLASS_DEVICE
    MatrixCoord GetBlockOffset(MatrixCoord blockIdx) const {
        MatrixCoord commBlockCoord{blockIdx.row(), 0};
        auto subBlockOffset = commBlockCoord * blockShape;

        return subBlockOffset;
    }

    template <typename CommOp>  CATLASS_DEVICE
    MatrixCoord GetRankOffset(MatrixCoord blockIdx) const {
        MatrixCoord commRankCoord;
        if constexpr (std::is_same<CommOp, ReduceScatterSchedule>::value) {
            commRankCoord = MatrixCoord{rankIdx, 0};
        }
        else if constexpr (std::is_same<CommOp, AllGatherSchedule>::value) {
            commRankCoord = MatrixCoord{blockIdx.column(), 0};
        }
        auto rankBlockOffset = commRankCoord * problemSizePerRank;
        return rankBlockOffset;
    }

    template <typename CommOp> CATLASS_DEVICE
    MatrixCoord GetBlockSize(MatrixCoord blockIdx) const {
        auto blockTileOffset = GetRankOffset<CommOp>(blockIdx) + GetBlockOffset(blockIdx);

        if (blockTileOffset.row() >= problemSize.row()) {
            return MatrixCoord{};
        }

        uint32_t mActual = (blockIdx.row() == (mLoops - 1)) ?
            (problemSizePerRank.row() - blockIdx.row() * blockShape.row()) : blockShape.row();
        if (blockTileOffset.row() + mActual >= problemSize.row()) {
            mActual = problemSize.row() - blockTileOffset.row();
        }

        return MatrixCoord{mActual, problemSizePerRank.column()};
    }
};

}  // namespace Catlass::Gemm::Block

   