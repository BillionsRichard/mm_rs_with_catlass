/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCOC_DGEMM_BLOCK_SWIZZLE_ALLGATHER_HPP
#define CATCOC_DGEMM_BLOCK_SWIZZLE_ALLGATHER_HPP

#pragma once

#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"

#include "catcoc/dist_coord.hpp"
namespace Catcoc::DGemm::Block {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

/// Threadblock swizzling function for GEMMs
template <uint32_t SWIZZLE_OFFSET = 1, uint32_t SWIZZLE_DIRECTION = 0>
    ///
    /// Data members
    ///
struct GemmBlockSwizzleAllGatherMesh {
    DistGemmCoord problemShape; // problemSize for total distributed Gemm
    DistGemmCoord loops;
    DistGemmCoord tileShape;
    ///
    /// Methods
    ///

    CATLASS_DEVICE
    GemmBlockSwizzleAllGatherMesh() = default;

    CATLASS_DEVICE
    GemmBlockSwizzleAllGatherMesh(DistGemmCoord const &problemShape_, DistGemmCoord const &tileShape_) :
        problemShape(problemShape_), tileShape(tileShape_)
    {
        loops = CeilDiv(problemShape, tileShape);
    }

    CATLASS_DEVICE
    GemmBlockSwizzleAllGatherMesh(DistGemmCoord const &problemShape_, MatrixCoord const &tileShapeMN_) :
        problemShape(problemShape_)
    {
        tileShape = Catlass::MakeCoord<uint32_t>(tileShapeMN_[0], tileShapeMN_[1], problemShape_[2], 1);
        loops = CeilDiv(problemShape, tileShape);
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const
    {
        return loops[0] * loops[1] * loops[2] * loops[3];
    }

    CATLASS_DEVICE
    DistGemmCoord GetBlockCoord(uint32_t loopIdx) const
    {
        uint32_t rows = loops[0] * loops[3];
        uint32_t cols = loops[1];
        uint32_t rowIdx{}, colIdx{};
        if constexpr (SWIZZLE_DIRECTION == 0) {
            uint32_t groupSize = SWIZZLE_OFFSET * cols;
            uint32_t groupIdx = loopIdx / groupSize;
            uint32_t groupOffset = loopIdx - groupIdx * groupSize;

            uint32_t inGroupRows = Min(SWIZZLE_OFFSET, rows - groupIdx * SWIZZLE_OFFSET);
            colIdx = groupOffset / inGroupRows;
            uint32_t inGroupRowIdx = groupOffset - colIdx * inGroupRows;
            rowIdx = groupIdx * SWIZZLE_OFFSET + inGroupRowIdx;
            if ((groupIdx & 0b1) == 1) {
                colIdx = cols - colIdx - 1;
            }
        } else if constexpr (SWIZZLE_DIRECTION == 1) {
            uint32_t groupSize = SWIZZLE_OFFSET * rows;
            uint32_t groupIdx = loopIdx / groupSize;
            uint32_t groupOffset = loopIdx - groupIdx * groupSize;
            uint32_t inGroupCols = Min(SWIZZLE_OFFSET, cols - groupIdx * SWIZZLE_OFFSET);
            rowIdx = groupOffset / inGroupCols;
            uint32_t inGroupColIdx = groupOffset - rowIdx * inGroupCols;
            colIdx = groupIdx * SWIZZLE_OFFSET + inGroupColIdx;
            if ((groupIdx & 0b1) == 1) {
                rowIdx = rows - rowIdx - 1;
            }
        }
        return {rowIdx % loops[0], colIdx, 0, rowIdx / loops[0]};
    }

    CATLASS_DEVICE
    DistGemmCoord GetBlockOffset(uint32_t loopIdx) const
    {
        return GetBlockCoord(loopIdx) * tileShape;
    }

    CATLASS_DEVICE
    DistGemmCoord GetActualBlockShapeByOffset(DistGemmCoord blockOffset) const
    {
        return Min(tileShape, problemShape - blockOffset);

        // 局部swizzle


    }

    CATLASS_DEVICE
    DistGemmCoord GetActualBlockShape(DistGemmCoord blockCoord)
    {
        auto blockOffset = blockCoord * tileShape;
        return GetActualBlockShapeByOffset(blockOffset);
    }
};

} // namespace Catcoc::DGemm::Block

#endif // CATCOC_DGEMM_BLOCK_SWIZZLE_ALLGATHER_HPP