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
#include "../utils/func_type.h"

#define KERNEL_P(NAME, TYPE)                                                            \
class kernel_##NAME##_p {                                                               \
public:                                                                                 \
    __aicore__ inline kernel_##NAME##_p() {}                                            \
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)                               \
    {                                                                                   \
        gva_gm = (__gm__ TYPE *)gva;                                                    \
        dev_gm = (__gm__ TYPE *)dev;                                                    \
                                                                                        \
        rank = smem_shm_get_global_rank();                                              \
        rank_size = smem_shm_get_global_rank_size();                                    \
    }                                                                                   \
    __aicore__ inline void Process()                                                    \
    {                                                                                   \
        shmem_##NAME##_p(gva_gm, *dev_gm, (rank + 1) % rank_size);                      \
    }                                                                                   \
private:                                                                                \
    __gm__ TYPE *gva_gm;                                                                \   
    __gm__ TYPE *dev_gm;                                                                \
                                                                                        \
    int64_t rank;                                                                       \
    int64_t rank_size;                                                                  \
}

SHMEM_FUNC_TYPE_KERNEL(KERNEL_P);

#define P_NUM_TEST(NAME, TYPE)                                                          \
extern "C" __global__ __aicore__ void p_##NAME##_num_test(GM_ADDR gva, GM_ADDR dev)     \
{                                                                                       \
    kernel_##NAME##_p op;                                                               \
    op.Init(gva, dev);                                                                  \
    op.Process();                                                                       \
}

SHMEM_FUNC_TYPE_KERNEL(P_NUM_TEST);

#define PUT_ONE_NUM_DO(NAME, TYPE)                                                          \
void put_##NAME##_one_num_do(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev)  \
{                                                                                           \
    p_##NAME##_num_test<<<block_dim, nullptr, stream>>>(gva, dev);                          \
}

SHMEM_FUNC_TYPE_KERNEL(PUT_ONE_NUM_DO);

#define KERNEL_G(NAME, TYPE)                                                            \
class kernel_##NAME##_g {                                                               \
public:                                                                                 \
    __aicore__ inline kernel_##NAME##_g() {}                                            \
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)                               \
    {                                                                                   \
        gva_gm = (__gm__ TYPE *)gva;                                                    \
        dev_gm = (__gm__ TYPE *)dev;                                                    \
                                                                                        \
        rank = smem_shm_get_global_rank();                                              \
        rank_size = smem_shm_get_global_rank_size();                                    \
    }                                                                                   \
    __aicore__ inline void Process()                                                    \
    {                                                                                   \
        TYPE val = shmem_##NAME##_g(gva_gm, (rank + 1) % rank_size);                    \
        *dev_gm = val;                                                                  \
    }                                                                                   \
private:                                                                                \
    __gm__ TYPE *gva_gm;                                                                \   
    __gm__ TYPE *dev_gm;                                                                \
                                                                                        \
    int64_t rank;                                                                       \
    int64_t rank_size;                                                                  \
}

SHMEM_FUNC_TYPE_KERNEL(KERNEL_G);

#define G_NUM_TEST(NAME, TYPE)                                                          \
extern "C" __global__ __aicore__ void g_##NAME##_num_test(GM_ADDR gva, GM_ADDR dev)     \
{                                                                                       \
    kernel_##NAME##_g op;                                                               \
    op.Init(gva, dev);                                                                  \
    op.Process();                                                                       \
}

SHMEM_FUNC_TYPE_KERNEL(G_NUM_TEST);

#define GET_ONE_NUM_DO(NAME, TYPE)                                                          \
void get_##NAME##_one_num_do(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev)  \
{                                                                                           \
    g_##NAME##_num_test<<<block_dim, nullptr, stream>>>(gva, dev);                          \
}

SHMEM_FUNC_TYPE_KERNEL(GET_ONE_NUM_DO);