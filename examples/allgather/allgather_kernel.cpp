/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

extern "C" __attribute__((always_inline)) inline __aicore__ void all_gather_origin(GM_ADDR input, GM_ADDR output, GM_ADDR gva, int elements, int copy_len, int idx)
{
    int64_t my_rank = shmem_my_pe();
    int64_t pe_size = shmem_n_pes();
    __gm__ int32_t* gva_gm = (__gm__ int32_t *)gva;
    __gm__ int32_t* sync_gm = (__gm__ int32_t *)gva + 128 * 1024 * 1024;

    __gm__ int32_t* input_gm = (__gm__ int32_t *)input;
    __gm__ int32_t* output_gm = (__gm__ int32_t *)output;
    AscendC::PipeBarrier<PIPE_ALL>();
    cube_guard();
    int aivNum = AscendC::GetBlockNum();
    int aivIndex = AscendC::GetBlockIdx() / 2;

    // 0-7 copy data to local symmetric mem, 8-15 copy remote data from symmetric mem.
    int corePerGroup = aivNum / 2;
    int corePerRank = corePerGroup / pe_size;
    int lenPerCore = copy_len / corePerGroup;

    // Only use one AIV Core.
    if (AscendC::GetSubBlockIdx() != 0) {
        return;
    }

    // Sync UB
    __ubuf__ int32_t* ctrlFlagsUB[16];
    for (int i = 0; i < corePerGroup; i++) {
        ctrlFlagsUB[i] = (__ubuf__ int32_t*)(0) + i * 128;
    }

    int sync_flag = 1024 * (idx + 1);

    // GM to SymmPtr
    if (aivIndex < corePerGroup) {
        shmem_put_int32_mem_nbi(gva_gm + aivIndex * lenPerCore, input_gm + aivIndex * lenPerCore, lenPerCore, my_rank);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);

        *ctrlFlagsUB[aivIndex] = sync_flag;
        AscendC::PipeBarrier<PIPE_ALL>();
        shmem_put_int32_mem_nbi(sync_gm + aivIndex * 128, ctrlFlagsUB[aivIndex], 1, my_rank);
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }

    // All Gather PingPong
    __ubuf__ int32_t* syncBuff = reinterpret_cast<__ubuf__ int32_t*>(uint64_t(0));
    __ubuf__ int32_t* tmpBuff1 = reinterpret_cast<__ubuf__ int32_t*>(uint64_t(4 * 1024));
    __ubuf__ int32_t* tmpBuff2 = reinterpret_cast<__ubuf__ int32_t*>(uint64_t(80 * 1024));
    uint32_t copy_ub_size = 16 * 1024;
    uint32_t copy_ub_num = copy_ub_size / sizeof(int32_t);

    int x = (aivIndex - corePerGroup) / corePerRank;

    int pingpongId = 0;
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int blockGroupIdx = 0; blockGroupIdx < corePerGroup; blockGroupIdx++)
    {
        shmem_signal_wait_until((__gm__ int32_t *)shmem_ptr(sync_gm + blockGroupIdx * 128, x), SHMEM_CMP_EQ, sync_flag);
        shmem_fence();
        int group_recv_offset = x * elements + blockGroupIdx * lenPerCore;
        int group_send_offset = blockGroupIdx * lenPerCore;

        int numTotal = lenPerCore;
        int send_offset = 0;
        int recv_offset = 0;
        for (int i = 0; numTotal > 0; i++) {
            AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
            __ubuf__ int32_t* buf = pingpongId == 0 ? tmpBuff1 : tmpBuff2;

            uint32_t copy_num = numTotal > copy_ub_num ? copy_ub_num : numTotal;

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            shmem_mte_get_mem_nbi(output_gm + group_recv_offset + recv_offset, gva_gm + group_send_offset + send_offset, buf, copy_ub_size, copy_num, x, EVENT_ID);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);

            send_offset += copy_num;
            recv_offset += copy_num;
            numTotal -= copy_num;
            pingpongId = 1 - pingpongId;
        }
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// all_gather
extern "C" __global__ __aicore__ void device_all_gather_test(GM_ADDR input, GM_ADDR output, GM_ADDR gva, int elements)
{
    int64_t max_gva_size = 128 * 1024 * 1024 / sizeof(int32_t);
    int times = (elements + max_gva_size - 1) / max_gva_size;
    int total_num = elements;
    for (int i = 0; i < times; i++) {
        int copy_len = total_num > max_gva_size ? max_gva_size : total_num;
        shmemx_barrier_all_vec();
        all_gather_origin(input + i * max_gva_size * sizeof(int32_t), output + i * max_gva_size * sizeof(int32_t), gva, elements, copy_len, i);
        total_num -= max_gva_size;
    }
}

void allgather_demo(uint32_t block_dim, void* stream, uint8_t* input, uint8_t* output, uint8_t* gva, int elements)
{
    device_all_gather_test<<<block_dim, nullptr, stream>>>(input, output, gva, elements);
}

#endif  // _KERNEL_ALLGATHER_