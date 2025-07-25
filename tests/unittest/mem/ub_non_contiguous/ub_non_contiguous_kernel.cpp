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
#include "smem_shm_aicore_base_api.h"

#include "shmem_api.h"
#include "../utils/func_type.h"

#define KERNEL_UB_PUT_NUM_NON_CONTIGUOUS(NAME, TYPE)                                                                                    \
    class kernel_ub_##NAME##_put_num_non_contiguous {                                                                                   \
    public:                                                                                                                             \
        __aicore__ inline kernel_ub_##NAME##_put_num_non_contiguous() {}                                                                \
        __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)                                                                           \
        {                                                                                                                               \
            gva_gm = (__gm__ TYPE*)gva;                                                                                                 \
            dev_gm = (__gm__ TYPE*)dev;                                                                                                 \
                                                                                                                                        \
            rank = shmem_my_pe();                                                                                                       \
            rank_size = shmem_n_pes();                                                                                                  \
                                                                                                                                        \
            /* set GM Buffer */                                                                                                         \
            src_gm.SetGlobalBuffer(dev_gm);                                                                                             \
            dst_gm.SetGlobalBuffer(gva_gm);                                                                                             \
                                                                                                                                        \
            /* 1x4096 Bytes Buffer */                                                                                                   \
            pipe.InitBuffer(buf_queue, 1, 4096);                                                                                        \
        }                                                                                                                               \
        __aicore__ inline void Process(int repeat, int length)                                                                          \
        {                                                                                                                               \
            AscendC::LocalTensor<TYPE> buf_tensor = buf_queue.AllocTensor<TYPE>();                                                      \
            __ubuf__ TYPE *buf = (__ubuf__ TYPE *)buf_tensor.address_.bufferAddr;                                                       \
            AscendC::DataCopy(buf_tensor, src_gm, repeat * length);                                                                     \
                                                                                                                                        \
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);                                                                 \
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);                                                                \
                                                                                                                                        \
            non_contiguous_copy_param copy_params;                                                                                      \
            /* Test all overloaded interfaces, divide task into 4 pieces. */                                                            \
            int task_repeat = repeat / 4;                                                                                               \
            copy_params.repeat = task_repeat / 2;   /* Only copy even lines. */                                                         \
            copy_params.length = length;                                                                                                \
            copy_params.src_ld = 2 * length;                                                                                            \
            copy_params.dst_ld = length;                                                                                                \
                                                                                                                                        \
            int src_offset = task_repeat * length;                                                                                      \
            int dst_offset = task_repeat / 2 * length;                                                                                  \
            shmem_mte_put_mem_nbi(dst_gm[dst_offset * 0], buf_tensor[src_offset * 0], copy_params, (rank + 1) % rank_size, EVENT_ID0);  \
            shmem_mte_put_mem_nbi(gva_gm + dst_offset * 1, buf + src_offset * 1, copy_params, (rank + 1) % rank_size, EVENT_ID0);       \
            shmem_put_##NAME##_mem_nbi(dst_gm[dst_offset * 2], buf_tensor[src_offset * 2], copy_params, (rank + 1) % rank_size);        \
            shmem_put_##NAME##_mem_nbi(gva_gm + dst_offset * 3, buf + src_offset * 3, copy_params, (rank + 1) % rank_size);             \
                                                                                                                                        \
            shmemx_barrier_all_vec();                                                                                                   \
            buf_queue.FreeTensor(buf_tensor);                                                                                           \
        }                                                                                                                               \
    private:                                                                                                                            \
        AscendC::TPipe pipe;                                                                                                            \
        AscendC::TQue<AscendC::TPosition::VECIN, 2> buf_queue;                                                                          \
                                                                                                                                        \
        AscendC::GlobalTensor<TYPE> src_gm, dst_gm;                                                                                     \
        __gm__ TYPE *gva_gm;                                                                                                            \
        __gm__ TYPE *dev_gm;                                                                                                            \
                                                                                                                                        \
        int64_t rank;                                                                                                                   \
        int64_t rank_size;                                                                                                              \
    } 

SHMEM_FUNC_TYPE_KERNEL(KERNEL_UB_PUT_NUM_NON_CONTIGUOUS);        

# define UB_PUT_NUM_NON_CONTIGUOUS_TEST(NAME, TYPE)                                                                                     \
    extern "C" __global__ __aicore__ void ub_##NAME##_put_non_contiguous_num_test(GM_ADDR gva, GM_ADDR dev, int repeat, int length)     \
    {                                                                                                                                   \
        kernel_ub_##NAME##_put_num_non_contiguous op;                                                                                   \
        op.Init(gva, dev);                                                                                                              \
        op.Process(repeat, length);                                                                                                     \
    }

SHMEM_FUNC_TYPE_KERNEL(UB_PUT_NUM_NON_CONTIGUOUS_TEST);

# define TEST_UB_NON_CONTIGUOUS_PUT(NAME, TYPE)                                                                                         \
    void test_ub_##NAME##_non_contiguous_put(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev, int repeat, int length)      \
    {                                                                                                                                   \
        ub_##NAME##_put_non_contiguous_num_test<<<block_dim, nullptr, stream>>>(gva, dev, repeat, length);                              \
    }

SHMEM_FUNC_TYPE_KERNEL(TEST_UB_NON_CONTIGUOUS_PUT);

#define KERNEL_UB_GET_NUM_NON_CONTIGUOUS(NAME, TYPE)                                                                                    \
    class kernel_ub_##NAME##_get_num_non_contiguous {                                                                                   \
    public:                                                                                                                             \
        __aicore__ inline kernel_ub_##NAME##_get_num_non_contiguous() {}                                                                \
        __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)                                                                           \
        {                                                                                                                               \
            gva_gm = (__gm__ TYPE *)gva;                                                                                                \
            dev_gm = (__gm__ TYPE *)dev;                                                                                                \
                                                                                                                                        \
            rank = shmem_my_pe();                                                                                                       \
            rank_size = shmem_n_pes();                                                                                                  \
                                                                                                                                        \
            /* set GM Buffer */                                                                                                         \
            src_gm.SetGlobalBuffer(gva_gm);                                                                                             \
            dst_gm.SetGlobalBuffer(dev_gm);                                                                                             \
                                                                                                                                        \
            /* 1x4096 Bytes Buffer */                                                                                                   \
            pipe.InitBuffer(buf_queue, 1, 4096);                                                                                        \
        }                                                                                                                               \
        __aicore__ inline void Process(int repeat, int length)                                                                          \
        {                                                                                                                               \
            AscendC::LocalTensor<TYPE> buf_tensor = buf_queue.AllocTensor<TYPE>();                                                      \
            __ubuf__ TYPE *buf = (__ubuf__ TYPE *)buf_tensor.address_.bufferAddr;                                                       \
                                                                                                                                        \
            non_contiguous_copy_param copy_params;                                                                                      \
            /* Test all overloaded interfaces, divide task into 4 pieces. */                                                            \
            int task_repeat = repeat / 4;                                                                                               \
            copy_params.repeat = task_repeat / 2;   /* Only copy even lines. */                                                         \
            copy_params.length = length;                                                                                                \
            copy_params.src_ld = 2 * length;                                                                                            \
            copy_params.dst_ld = length;                                                                                                \
                                                                                                                                        \
            int src_offset = task_repeat * length;                                                                                      \
            int dst_offset = task_repeat / 2 * length;                                                                                  \
            shmem_mte_get_mem_nbi(buf + dst_offset * 0, gva_gm + src_offset * 0, copy_params, (rank + 1) % rank_size, EVENT_ID0);       \
            shmem_mte_get_mem_nbi(buf_tensor[dst_offset * 1], src_gm[src_offset * 1], copy_params, (rank + 1) % rank_size, EVENT_ID0);  \
            shmem_get_##NAME##_mem_nbi(buf + dst_offset * 2, gva_gm + src_offset * 2, copy_params, (rank + 1) % rank_size);             \
            shmem_get_##NAME##_mem_nbi(buf_tensor[dst_offset * 3], src_gm[src_offset * 3], copy_params, (rank + 1) % rank_size);        \
                                                                                                                                        \
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);                                                                 \
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);                                                                \
                                                                                                                                        \
            AscendC::DataCopy(dst_gm, buf_tensor, repeat * length);                                                                     \
            shmemx_barrier_all_vec();                                                                                                   \
            buf_queue.FreeTensor(buf_tensor);                                                                                           \
        }                                                                                                                               \
    private:                                                                                                                            \
        AscendC::TPipe pipe;                                                                                                            \
        AscendC::TQue<AscendC::TPosition::VECIN, 2> buf_queue;                                                                          \
        AscendC::GlobalTensor<TYPE> src_gm, dst_gm;                                                                                     \
        __gm__ TYPE *gva_gm;                                                                                                            \
        __gm__ TYPE *dev_gm;                                                                                                            \
                                                                                                                                        \
        int64_t rank;                                                                                                                   \
        int64_t rank_size;                                                                                                              \
    } 

SHMEM_FUNC_TYPE_KERNEL(KERNEL_UB_GET_NUM_NON_CONTIGUOUS);

#define UB_GET_NON_CONTIGUUS_NUM_TEST(NAME, TYPE)                                                                                       \
    extern "C" __global__ __aicore__ void ub_##NAME##_get_non_contiguous_num_test(GM_ADDR gva, GM_ADDR dev, int repeat, int length)     \
    {                                                                                                                                   \
        kernel_ub_##NAME##_get_num_non_contiguous op;                                                                                   \
        op.Init(gva, dev);                                                                                                              \
        op.Process(repeat, length);                                                                                                     \
    }

SHMEM_FUNC_TYPE_KERNEL(UB_GET_NON_CONTIGUUS_NUM_TEST);

#define TEST_UB_NON_CONTIGUOUS_GET(NAME, TYPE)                                                                                          \
    void test_ub_##NAME##_non_contiguous_get(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev, int repeat, int length)      \
    {                                                                                                                                   \
        ub_##NAME##_get_non_contiguous_num_test<<<block_dim, nullptr, stream>>>(gva, dev, repeat, length);                              \
    }

SHMEM_FUNC_TYPE_KERNEL(TEST_UB_NON_CONTIGUOUS_GET);