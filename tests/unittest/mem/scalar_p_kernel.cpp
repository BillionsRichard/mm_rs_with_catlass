/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "shmem_api.h"

class kernel_p {
public:
    __aicore__ inline kernel_p() {}
    __aicore__ inline void Init(GM_ADDR gva, float val)
    {
        gva_gm = (__gm__ float *)gva;
        value = val;

        rank = smem_shm_get_global_rank();
        rank_size = smem_shm_get_global_rank_size();
    }
    __aicore__ inline void Process()
    {
        shmem_float_p(gva_gm, value, (rank + 1) % rank_size);
    }
private:
    __gm__ float *gva_gm;
    float value;

    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void p_num_test(GM_ADDR gva, float val)
{
    kernel_p op;
    op.Init(gva, val);
    op.Process();
}

void put_one_num_do(uint32_t block_dim, void* stream, uint8_t* gva, float val)
{
    p_num_test<<<block_dim, nullptr, stream>>>(gva, val);
}