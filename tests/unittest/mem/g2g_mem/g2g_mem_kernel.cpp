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

const int nmem = 16;

#define KERNEL_G2G_PUT_NUM(NAME, TYPE)                                                                                              \
    class kernel_g2g_##NAME##_put_num {                                                                                             \
    public:                                                                                                                         \
        __aicore__ inline kernel_g2g_##NAME##_put_num() {}                                                                          \
        __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)                                                                       \
        {                                                                                                                           \
            gva_gm = (__gm__ TYPE *)gva;                                                                                            \
            dev_gm = (__gm__ TYPE *)dev;                                                                                            \
                                                                                                                                    \
            /* set GM Buffer */                                                                                                     \
            src_gm.SetGlobalBuffer(dev_gm);                                                                                         \
            dst_gm.SetGlobalBuffer(gva_gm);                                                                                         \
                                                                                                                                    \
            rank = shmem_my_pe();                                                                                                   \
            rank_size = shmem_n_pes();                                                                                              \
                                                                                                                                    \
            /* 1x4096 Bytes Buffer */                                                                                               \
            pipe.InitBuffer(buf_queue, 1, 4096);                                                                                    \
        }                                                                                                                           \
        __aicore__ inline void Process()                                                                                            \
        {                                                                                                                           \
            AscendC::LocalTensor<TYPE> buf_tensor = buf_queue.AllocTensor<TYPE>();                                                  \
            shmem_mte_put_mem_nbi(dst_gm, src_gm, buf_tensor, rank_size / 2 * nmem, rank, EVENT_ID0);                               \
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);                                                             \
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);                                                            \
            shmem_put_##NAME##_mem_nbi(dst_gm[rank_size / 2 * nmem], src_gm[rank_size / 2 * nmem], rank_size / 2 * nmem, rank);     \
            shmemx_barrier_all_vec();                                                                                               \
            buf_queue.FreeTensor(buf_tensor);                                                                                       \
        }                                                                                                                           \
    private:                                                                                                                        \
        AscendC::TPipe pipe;                                                                                                        \
        AscendC::TQue<AscendC::TPosition::VECIN, 2> buf_queue;                                                                      \
        AscendC::GlobalTensor<TYPE> src_gm, dst_gm;                                                                                 \
                                                                                                                                    \
        __gm__ TYPE *gva_gm;                                                                                                        \
        __gm__ TYPE *dev_gm;                                                                                                        \
                                                                                                                                    \
        int64_t rank;                                                                                                               \
        int64_t rank_size;                                                                                                          \
    }

SHMEM_FUNC_TYPE_KERNEL(KERNEL_G2G_PUT_NUM);

#define PUT_G2G_NUM_TEST(NAME, TYPE)                                                            \
    extern "C" __global__ __aicore__ void put_g2g_##NAME##_num_test(GM_ADDR gva, GM_ADDR dev)   \
    {                                                                                           \
        kernel_g2g_##NAME##_put_num op;                                                         \
        op.Init(gva, dev);                                                                      \
        op.Process();                                                                           \
    }

SHMEM_FUNC_TYPE_KERNEL(PUT_G2G_NUM_TEST);

#define TEST_G2G_PUT(NAME, TYPE)                                                                \
    void test_g2g_##NAME##_put(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev)    \
    {                                                                                           \
        put_g2g_##NAME##_num_test<<<block_dim, nullptr, stream>>>(gva, dev);                    \
    }

SHMEM_FUNC_TYPE_KERNEL(TEST_G2G_PUT);

#define KERNEL_G2G_GET_NUM(NAME, TYPE)                                                                                                  \
    class kernel_g2g_##NAME##_get_num {                                                                                                 \
    public:                                                                                                                             \
        __aicore__ inline kernel_g2g_##NAME##_get_num() {}                                                                              \
        __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)                                                                           \
        {                                                                                                                               \
            gva_gm = (__gm__ TYPE *)gva;                                                                                                \
            dev_gm = (__gm__ TYPE *)dev;                                                                                                \
                                                                                                                                        \
            /* set GM Buffer */                                                                                                         \
            src_gm.SetGlobalBuffer(dev_gm);                                                                                             \
            dst_gm.SetGlobalBuffer(gva_gm);                                                                                             \
                                                                                                                                        \
            rank = shmem_my_pe();                                                                                                       \
            rank_size = shmem_n_pes();                                                                                                  \
                                                                                                                                        \
            /* 1x4096 Bytes Buffer */                                                                                                   \
            pipe.InitBuffer(buf_queue, 1, 4096);                                                                                        \
        }                                                                                                                               \
        __aicore__ inline void Process()                                                                                                \
        {                                                                                                                               \
            AscendC::LocalTensor<TYPE> buf_tensor = buf_queue.AllocTensor<TYPE>();                                                      \
                                                                                                                                        \
            for (int i = 0; i < rank_size / 2; i++) {                                                                                   \
                shmem_mte_get_mem_nbi(src_gm[nmem * i], dst_gm, buf_tensor, nmem, i % rank_size, EVENT_ID0);                            \
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);                                                             \
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);                                                            \
            }                                                                                                                           \
                                                                                                                                        \
            for (int i = rank_size / 2; i < rank_size; i++) {                                                                           \
                shmem_get_##NAME##_mem_nbi(src_gm[nmem * i], dst_gm, nmem, i % rank_size);                                              \
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);                                                             \
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);                                                            \
            }                                                                                                                           \
                                                                                                                                        \
            shmemx_barrier_all_vec();                                                                                                   \
            buf_queue.FreeTensor(buf_tensor);                                                                                           \
        }                                                                                                                               \
    private:                                                                                                                            \
        AscendC::TPipe pipe;                                                                                                            \
        AscendC::TQue<AscendC::TPosition::VECIN, 2> buf_queue;                                                                          \
        __gm__ TYPE *gva_gm;                                                                                                            \
        __gm__ TYPE *dev_gm;                                                                                                            \
        AscendC::GlobalTensor<TYPE> src_gm, dst_gm;                                                                                     \
                                                                                                                                        \
        int64_t rank;                                                                                                                   \
        int64_t rank_size;                                                                                                              \
    }

SHMEM_FUNC_TYPE_KERNEL(KERNEL_G2G_GET_NUM);

#define GET_G2G_NUM_TEST(NAME,TYPE)                                                             \
    extern "C" __global__ __aicore__ void get_g2g_##NAME##_num_test(GM_ADDR gva, GM_ADDR dev)   \
    {                                                                                           \
        kernel_g2g_##NAME##_get_num op;                                                         \
        op.Init(gva, dev);                                                                      \
        op.Process();                                                                           \
    }

SHMEM_FUNC_TYPE_KERNEL(GET_G2G_NUM_TEST);

#define TEST_G2G_GET(NAME, TYPE)                                                                \
    void test_g2g_##NAME##_get(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev)    \
    {                                                                                           \
        get_g2g_##NAME##_num_test<<<block_dim, nullptr, stream>>>(gva, dev);                    \
    }

SHMEM_FUNC_TYPE_KERNEL(TEST_G2G_GET);