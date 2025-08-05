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

template <uint32_t SWIZZLE_DIRECTION_ = 0, bool IS_DETERMINISTIC_ = false>
struct BlockCommSwizzle {
    static constexpr uint32_t SWIZZLE_DIRECTION = SWIZZLE_DIRECTION_;
    static constexpr uint32_t IS_DETERMINISTIC = IS_DETERMINISTIC_;

    static_assert((IS_DETERMINISTIC && SWIZZLE_DIRECTION == 0) || !IS_DETERMINISTIC, 
        "Deterministic calculation requires that the swizzle direction be 0.");

    uint32_t rankSize, curRankIdx;
    MatrixCoord problemSize;
    uint32_t dataLoopsInRank, rankLoops;

    uint32_t nStride;

    uint32_t swizzleOffset;
    MatrixCoord coreSplit;
    MatrixCoord blockShape;

    CATLASS_DEVICE
    BlockCommSwizzle() {}

    template <detail::CopyMode CopyMode_, detail::CopyDirect CopyDirect_>
    CATLASS_DEVICE
    BlockCommSwizzle(uint32_t curRankIdx_, uint32_t rankSize_, MatrixCoord const &coreSplit_,
        MatrixCoord const &problemSize_, MatrixCoord const &blockShape_)
        : curRankIdx(curRankIdx_), rankSize(rankSize_), coreSplit(coreSplit_),
          problemSize(problemSize_), blockShape(blockShape_)
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
        rankLoops = rankSize;
        nStride = rankSize / coreSplit.column();
        MatrixCoord dataLoopsMx = CeilDiv(problemSize, blockShape);
        if constexpr (CopyMode_ == detail::CopyMode::P2P) {
            dataLoopsInRank = dataLoopsMx.row() * dataLoopsMx.column();
        } else {
            dataLoopsInRank = CeilDiv(dataLoopsMx.row() * dataLoopsMx.column(), rankSize);
        }
    }

    CATLASS_DEVICE
    BlockCommSwizzle(uint32_t curRankIdx_, uint32_t rankSize_, MatrixCoord const &coreSplit_,
        MatrixCoord const &problemSize_, MatrixCoord const &blockShape_, uint32_t dataLoopsInRank_)
        : curRankIdx(curRankIdx_), rankSize(rankSize_), coreSplit(coreSplit_),
          problemSize(problemSize_), blockShape(blockShape_), dataLoopsInRank(dataLoopsInRank_)
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
        rankLoops = rankSize;
        nStride = rankSize / coreSplit.column();
    }

    CATLASS_DEVICE
    BlockCommSwizzle(uint32_t curRankIdx_, uint32_t rankSize_, MatrixCoord const &coreSplit_)
        : curRankIdx(curRankIdx_), rankSize(rankSize_), coreSplit(coreSplit_)
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
        rankLoops = rankSize;
        nStride = rankSize / coreSplit.column();
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoop() const
    {
        if constexpr (IS_DETERMINISTIC) {
            return RoundUp<uint32_t>(dataLoopsInRank, coreSplit.row()) * rankLoops;
        } else {
            return dataLoopsInRank * rankLoops;
        }
    }

    CATLASS_DEVICE
    void Update(MatrixCoord const &problemSize_, MatrixCoord const &blockShape_, uint32_t dataLoopsInRank_) {
        problemSize = problemSize_;
        blockShape = blockShape_;
        dataLoopsInRank = dataLoopsInRank_;
    }

    template <detail::CopyMode CopyMode_, detail::CopyDirect CopyDirect_>
    CATLASS_DEVICE
    void Update(MatrixCoord const &problemSize_, MatrixCoord const &blockShape_) {
        problemSize = problemSize_;
        blockShape = blockShape_;
        MatrixCoord dataLoopsMx = CeilDiv(problemSize, blockShape);
        if constexpr (CopyMode_ == detail::CopyMode::P2P) {
            dataLoopsInRank = dataLoopsMx.row() * dataLoopsMx.column();
        } else {
            dataLoopsInRank = CeilDiv(dataLoopsMx.row() * dataLoopsMx.column(), rankSize);
        }
    }

    CATLASS_DEVICE
    uint32_t GetRealCore() const
    {
        return coreSplit.row() * coreSplit.column();
    }

    CATLASS_DEVICE
    MatrixCoord GetBlockIdx(uint32_t taskIdx) const {
        uint32_t innerIdx = taskIdx % GetCoreLoop();
        if constexpr (SWIZZLE_DIRECTION == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(dataLoopsInRank, swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * rankLoops);
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * rankLoops);

            uint32_t nRow = swizzleOffset;
            if constexpr (!IS_DETERMINISTIC) {
                if (tileBlockIdx == tileBlockLoop - 1) {
                    nRow = dataLoopsInRank - swizzleOffset * tileBlockIdx;
                }
            }
            uint32_t dataIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nRow;
            uint32_t rankIdx = inTileBlockIdx / nRow;
            
            rankIdx = (rankIdx * nStride) % rankLoops + (rankIdx * nStride) / rankLoops;
            rankIdx = (rankIdx + dataIdx) % rankLoops;

            return MatrixCoord{dataIdx, rankIdx};
        } else if (SWIZZLE_DIRECTION == 1) { // Nz
            uint32_t tileBlockLoop = CeilDiv(rankLoops, swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * dataLoopsInRank);
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * dataLoopsInRank);

            uint32_t nCol = swizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = rankLoops - swizzleOffset * tileBlockIdx;
            }
            uint32_t dataIdx = inTileBlockIdx / nCol;
            uint32_t rankIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nCol;
            
            rankIdx = (rankIdx * nStride) % rankLoops + (rankIdx * nStride) / rankLoops;
            rankIdx = (rankIdx + dataIdx) % rankLoops;

            return MatrixCoord{dataIdx, rankIdx};
        }
        return MatrixCoord{};
    }

    template <detail::CopyMode CopyMode_, detail::CopyDirect CopyDirect_>
    CATLASS_DEVICE
    MatrixCoord GetBlockOffset(MatrixCoord blockIdx, layout::AffineRankN<3> layoutC) const {
        uint32_t dataIdx = blockIdx.row();
        uint32_t rankIdx;
        if constexpr ((CopyMode_ == detail::CopyMode::Scatter && CopyDirect_ == detail::CopyDirect::Get)
                      || (CopyMode_ == detail::CopyMode::Gather && CopyDirect_ == detail::CopyDirect::Put)) {
            rankIdx = curRankIdx;
        } else if constexpr ((CopyMode_ == detail::CopyMode::Scatter && CopyDirect_ == detail::CopyDirect::Put)
                           || (CopyMode_ == detail::CopyMode::Gather && CopyDirect_ == detail::CopyDirect::Get)) {
            rankIdx = blockIdx.column();
        }
        return MatrixCoord{layoutC(Catlass::MakeCoord<int>(rankIdx, dataIdx, 0)), 0};
    }

    template <detail::CopyMode CopyMode_, detail::CopyDirect CopyDirect_>
    CATLASS_DEVICE
    MatrixCoord GetActualBlockShape(MatrixCoord blockIdx, layout::AffineRankN<3> layoutC) const
    {
        if (blockIdx.row() >= dataLoopsInRank) {
            return MatrixCoord{};
        }
        auto blockOffset = GetBlockOffset<CopyMode_, CopyDirect_>(blockIdx, layoutC);
        auto residue = problemSize - Min<uint32_t, 2>(problemSize, blockOffset);
        auto actualBlockShape = Min(blockShape, residue);
        return actualBlockShape;
    }
};

}  // namespace Catcoc::CommEpilogue::Block


#endif  // CATCOC_COMM_EPILOGUE_BLOCK_SWIZZLE_HPP