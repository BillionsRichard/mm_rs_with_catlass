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

namespace Catcoc::DGemm::Block {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

/// Threadblock swizzling function for GEMMs
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0, uint32_t BufferNum = 1>
struct GemmIdentityBlockSwizzleAllGather :
    public Catlass::Gemm::Block::GemmIdentityBlockSwizzle<SwizzleOffset, SwizzleDirection> {

    using Base = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<SwizzleOffset, SwizzleDirection>;
    ///
    /// Data members
    ///

    GemmCoord problemShape; // problemSize for total distributed Gemm
    MatrixCoord tileMN;
    MatrixCoord loopsMN;
    uint32_t rankSize;
    uint32_t mLoopsPerComm;

    ///
    /// Methods
    ///

    CATLASS_DEVICE
    GemmIdentityBlockSwizzleAllGather() {}

    CATLASS_DEVICE
    GemmIdentityBlockSwizzleAllGather(uint32_t rankSize_, uint32_t pValue_, GemmCoord const &problemShape_,
        MatrixCoord const &tileMN_)
        : Base(problemShape_, 
            tileMN_),
          rankSize(rankSize_), mLoopsPerComm(pValue_ * rankSize_), 
          problemShape(GemmCoord{problemShape_.m() * rankSize_, problemShape_.n(), problemShape_.k()}),
          tileMN(tileMN_)
    {
        loopsMN = Base::loopsMN * MatrixCoord{rankSize, 1};
    }


    CATLASS_DEVICE
    uint32_t GetBatchIdx(uint32_t taskIdx) const 
    {
        return Base::GetBatchIdx(taskIdx / rankSize);
    }

    CATLASS_DEVICE
    uint32_t GetMLoops() const 
    {
        return loopsMN.row();
    }

    CATLASS_DEVICE
    uint32_t GetNLoops() const 
    {
        return loopsMN.column();
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops(uint32_t batchCount = 1) const
    {
        return loopsMN.row() * loopsMN.column() * batchCount;
    }

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t taskIdx) 
    {
        uint32_t commCount = CeilDiv(loopsMN.row(), mLoopsPerComm);
        uint32_t commIdx = taskIdx / (mLoopsPerComm * Base::loopsMN.column());
        uint32_t inCommIdx = taskIdx % (mLoopsPerComm * Base::loopsMN.column());

        auto actualMLoops = 
            (commIdx == commCount - 1) ? loopsMN.row() - commIdx * mLoopsPerComm : mLoopsPerComm;
        auto actualMLoopsPerRank = actualMLoops / rankSize;

        // 局部swizzle
        auto tmpMNLoops = Base::loopsMN;
        Base::loopsMN = MatrixCoord{actualMLoops, tmpMNLoops.column()};
        auto coord = Base::GetBlockCoord(inCommIdx);
        Base::loopsMN = tmpMNLoops;

        uint32_t rankIdx = coord.m() / actualMLoopsPerRank;
        uint32_t m = rankIdx * Base::loopsMN.row() + commIdx * (mLoopsPerComm / rankSize) + coord.m() % actualMLoopsPerRank;
        
        return GemmCoord{m, coord.n(), coord.k()};
    }

    CATLASS_DEVICE
    GemmCoord GetActualBlockShape(GemmCoord blockTileOffset) 
    {
        uint32_t mIdxInRank = blockTileOffset.m() % Base::loopsMN.row();
        GemmCoord blockTileOffsetInRank(mIdxInRank, blockTileOffset.n(), blockTileOffset.k());
        return Base::GetActualBlockShape(blockTileOffsetInRank);
    }
};

}  // namespace Catcoc::DGemm::Block

#endif  // CATCOC_DGEMM_BLOCK_SWIZZLE_ALLGATHER_HPP