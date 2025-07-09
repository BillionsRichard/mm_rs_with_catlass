/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem_api.h"

// all_gather简易实现
extern "C" __global__ __aicore__ void device_all_gather_test(GM_ADDR gva, int elements)
{
    int64_t my_rank = shmem_my_pe();
    int64_t pe_size = shmem_n_pes();
    __gm__ int32_t* gva_gm = (__gm__ int32_t *)gva;
    AscendC::PipeBarrier<PIPE_ALL>();
    // All Gather
    for (int i = 0; i < pe_size; i++) {
        shmem_put_int32_mem_nbi(gva_gm + elements * my_rank, gva_gm + elements * my_rank, elements, i);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    shmemx_barrier_all_vec();
}

void allgather_demo(uint32_t block_dim, void* stream, uint8_t* gva, int elements)
{
    device_all_gather_test<<<block_dim, nullptr, stream>>>(gva, elements);
}