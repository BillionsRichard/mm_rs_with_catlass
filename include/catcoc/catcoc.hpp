/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCOC_CATLASS_HPP
#define CATCOC_CATLASS_HPP

#include <kernel_operator.h>

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"

namespace Catcoc {

template <typename Index, typename LongIndex, int RANK>
CATLASS_HOST_DEVICE
LongIndex Dot(Catlass::Coord<RANK, Index> const &coord, Catlass::Coord<RANK, LongIndex> const &stride,
    LongIndex accumulator = {})
{
    for (int i = 0; i < RANK; ++i) {
        accumulator += static_cast<LongIndex>(coord[i]) * stride[i];
    }
    return accumulator;
}

template <class T>
CATLASS_HOST_DEVICE constexpr
T Min(T const &lhs, T const &rhs)
{
    return (lhs < rhs) ? lhs : rhs;
}

template <typename Index, int RANK>
CATLASS_HOST_DEVICE constexpr
auto Min(Catlass::Coord<RANK, Index> const &lhs, Catlass::Coord<RANK, Index> const &rhs)
{
    Catlass::Coord<RANK, Index> result;
    for (int i = 0; i < RANK; ++i) {
        result[i] = Min(lhs[i], rhs[i]);
    }
    return result;
}

namespace layout {

template <int RANK_>
struct AffineRankN {
    static int const RANK = RANK_;
    using Index = int32_t;
    using LongIndex = int64_t;
    using TensorCoord = Catlass::Coord<RANK, Index>;
    using Stride = Catlass::Coord<RANK, LongIndex>;

private:
    Stride stride_;

public:
    CATLASS_HOST_DEVICE
    AffineRankN(Stride const &stride = Stride()) : stride_(stride) {}

    CATLASS_HOST_DEVICE
    static AffineRankN Packed(TensorCoord const &extent)
    {
        AffineRankN layout;
        layout.stride_[RANK - 1] = 1;

        for (int i = RANK - 1; i > 0; --i) {
            layout.stride_[i - 1] = layout.stride_[i] * extent[i];
        }

        return layout;
    }

    CATLASS_HOST_DEVICE
    LongIndex operator()(TensorCoord const &coord) const
    {
        return Dot(coord, stride_);
    }
};

};

}  // namespace Catcoc

#endif  // CATCOC_CATLASS_HPP