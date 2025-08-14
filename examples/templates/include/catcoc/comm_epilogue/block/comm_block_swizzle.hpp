/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCOC_COMM_EPILOGUE_BLOCK_SWIZZLE_HPP
#define CATCOC_COMM_EPILOGUE_BLOCK_SWIZZLE_HPP

#include "catcoc/catcoc.hpp"
#include "catcoc/detail/remote_copy_type.hpp"

// from catlass
#include "catlass/detail/alignment.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catcoc::CommEpilogue::Block {

using Catlass::MatrixCoord;

template<uint32_t SWIZZLE_DIRECTION_ = 0, bool IS_DETERMINISTIC_ = false>
struct BlockCommSwizzle {
    static constexpr uint32_t SWIZZLE_DIRECTION = SWIZZLE_DIRECTION_;
    static constexpr bool IS_DETERMINISTIC = IS_DETERMINISTIC_;

    static_assert((IS_DETERMINISTIC && SWIZZLE_DIRECTION == 0) || !IS_DETERMINISTIC,
        "Deterministic calculation requires that the swizzle direction be 0.");
    
    uint32_t rankSize, rankIdx;
    MatrixCoord problemSize;
    MatrixCoord problemSizePerRank;
    uint32_t mLoops, nLoops;

    uint32_t nStride;

    uint32_t swizzleOffset;
    MatrixCoord coreSplit;
    MatrixCoord blockShape;

    CATLASS_DEVICE
    BlockCommSwizzle() {}

    CATLASS_DEVICE
    BlockCommSwizzle(uint32_t rankIdx_, uint32_t rankSize_,
        MatrixCoord const &blockShape_, MatrixCoord const &coreSplit_) 
        : rankIdx(rankIdx_), rankSize(rankSize_), blockShape(blockShape_), coreSplit(coreSplit_)
    {
        // deterministic calculation does not allow npu split
        if constexpr (IS_DETERMINISTIC) {
            coreSplit = MatrixCoord{coreSplit.row() * coreSplit.column(), 1};
        }

        if constexpr (SWIZZLE_DIRECTION == 0) {
            swizzleOffset = coreSplit.row();
        } else {
            swizzleOffset = coreSplit.column();
        }
        nLoops = rankSize;
        nStride = rankSize / coreSplit.column();
    }
    
    CATLASS_DEVICE
    uint32_t GetCoreLoop() const
    {   
        if constexpr (IS_DETERMINISTIC) {
            return RoundUp<uint32_t>(mLoops, coreSplit.row()) * nLoops;
        } else {
            return mLoops * nLoops;
        }
    }

    template <detail::CopyMode CopyMode_, bool Align=false> CATLASS_DEVICE
    void SetProblemSize(MatrixCoord problemSize_)
    {
        problemSize = problemSize_;
        MatrixCoord commRankCount{rankSize, 1};
        if constexpr (CopyMode_ == detail::CopyMode::P2P) {
            problemSizePerRank = problemSize;
        } else {
            problemSizePerRank = CeilDiv(problemSize, commRankCount);
            if constexpr (Align) {
                problemSizePerRank =
                    {RoundUp<uint32_t>(problemSizePerRank.row(), blockShape.row()), problemSizePerRank.column()};
            }
        }
        mLoops = CeilDiv(problemSizePerRank.row(), blockShape.row());
    }

    CATLASS_DEVICE
    uint32_t GetRankStride() const
    {
        return problemSizePerRank.row();
    }

    CATLASS_DEVICE
    uint32_t GetRealCore() const
    {
        return coreSplit.row() * coreSplit.column();
    }

    CATLASS_DEVICE
    MatrixCoord GetBlockIdx(uint32_t taskIdx) const {
        uint32_t innerIdx = taskIdx % GetCoreLoop();
        if (SWIZZLE_DIRECTION == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(mLoops, swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * nLoops);
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * nLoops);

            uint32_t nRow = swizzleOffset;
            if constexpr (!IS_DETERMINISTIC) {
                if (tileBlockIdx == tileBlockLoop - 1) {
                    nRow = mLoops - swizzleOffset * tileBlockIdx;
                }
            }
            uint32_t mIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;

            nIdx = (nIdx * nStride) % nLoops + (nIdx * nStride) / nLoops;
            nIdx = (nIdx + mIdx) % nLoops;

            return MatrixCoord{mIdx, nIdx};
        } else if (SWIZZLE_DIRECTION == 1) { // Nz
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

    template <detail::CopyMode CopyMode_, detail::CopyDirect CopyDirect_> 
    CATLASS_DEVICE
    MatrixCoord GetBlockOffset(MatrixCoord blockIdx) const {
        MatrixCoord rankCoord;

        if constexpr ((CopyMode_ == detail::CopyMode::Scatter && CopyDirect_ == detail::CopyDirect::Get)
            ||(CopyMode_ == detail::CopyMode::Gather && CopyDirect_ == detail::CopyDirect::Put)) {
            rankCoord = MatrixCoord{rankIdx, 0};
        }
        else if constexpr ((CopyMode_ == detail::CopyMode::Scatter && CopyDirect_ == detail::CopyDirect::Put)
            ||(CopyMode_ == detail::CopyMode::Gather && CopyDirect_ == detail::CopyDirect::Get)) {
            rankCoord = MatrixCoord{blockIdx.column(), 0};
        }
        return rankCoord * problemSizePerRank + MatrixCoord{blockIdx.row(), 0} * blockShape;
    }

    template <detail::CopyMode CopyMode_, detail::CopyDirect CopyDirect_> 
    CATLASS_DEVICE
    MatrixCoord GetBlockOffset(MatrixCoord blockIdx, MatrixCoord rankStride) const {
        uint32_t rank;

        if constexpr ((CopyMode_ == detail::CopyMode::Scatter && CopyDirect_ == detail::CopyDirect::Get)
            ||(CopyMode_ == detail::CopyMode::Gather && CopyDirect_ == detail::CopyDirect::Put)) {
            rank = rankIdx;
        }
        else if constexpr ((CopyMode_ == detail::CopyMode::Scatter && CopyDirect_ == detail::CopyDirect::Put)
            ||(CopyMode_ == detail::CopyMode::Gather && CopyDirect_ == detail::CopyDirect::Get)) {
            rank = blockIdx.column();
        }
        auto layoutRankLogicShape = Catlass::MakeCoord<int64_t>(rankStride.row(), rankStride.column());
        auto layoutRank = layout::AffineRankN<2>(layoutRankLogicShape);
        auto offset = layoutRank(Catlass::MakeCoord<int>(blockIdx.row(), rank));
        return MatrixCoord{offset, 0};
    }

    template <detail::CopyMode CopyMode_, detail::CopyDirect CopyDirect_> 
    CATLASS_DEVICE
    MatrixCoord GetActualBlockShape(MatrixCoord blockIdx) const {
        if (blockIdx.row() >= mLoops) {
            return MatrixCoord{};
        }
        auto blockOffset = GetBlockOffset<CopyMode_, CopyDirect_>(blockIdx);
        auto residue = problemSize - Min<uint32_t, 2>(problemSize, blockOffset);
        auto actualBlockShape = Min(blockShape, residue);
        return actualBlockShape;
    }
};

}  // namespace Catcoc::CommEpilogue::Block

#endif // CATCOC_COMM_EPILOGUE_BLOCK_SWIZZLE_HPP