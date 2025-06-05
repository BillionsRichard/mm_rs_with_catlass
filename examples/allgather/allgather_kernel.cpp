/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _KERNEL_ALLGATHER_
#define _KERNEL_ALLGATHER_

#include "kernel_operator.h"
#include "shmem_api.h"

// 纯vec不能全核同步，需添加cube逻辑
SHMEM_DEVICE void cube_guard() 
{
    using namespace AscendC;

#ifdef __DAV_C220_CUBE__
    LocalTensor<float> result;
    result.address_.logicPos = (uint8_t)TPosition::CO1;
    result.InitBuffer(0, 256);
    
    LocalTensor<half> left;
    left.address_.logicPos = (uint8_t)TPosition::A2;
    left.InitBuffer(0, 256);

    LocalTensor<half> right;
    right.address_.logicPos = (uint8_t)TPosition::B2;
    right.InitBuffer(0, 256);

    MmadParams param;
    param.m = 16;
    param.n = 16;
    param.k = 16;

    Mmad<float, half, half>(result, left, right, param);
#endif
}

// all_gather简易实现
extern "C" __global__ __aicore__ void device_all_gather_test(GM_ADDR gva, int elements)
{
    int64_t my_rank = shmem_my_pe();
    int64_t pe_size = shmem_n_pes();
    __gm__ int32_t* gva_gm = (__gm__ int32_t *)gva;
    AscendC::PipeBarrier<PIPE_ALL>();
    cube_guard();
    // All Gather
    for (int i = 0; i < pe_size; i++) {
        shmem_put_int32_mem_nbi(gva_gm + elements * my_rank, gva_gm + elements * my_rank, elements, i);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    shmem_barrier_all();
}

void allgather_demo(uint32_t block_dim, void* stream, uint8_t* gva, int elements)
{
    device_all_gather_test<<<block_dim, nullptr, stream>>>(gva, elements);
}

#endif  // _KERNEL_ALLGATHER_