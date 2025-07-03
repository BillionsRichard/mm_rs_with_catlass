/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>

#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem_api.h"

int g_npus = 8;
const char *ipport;
int f_rank = 0;
int f_npu = 0;

using namespace AscendC;

constexpr int64_t MEM_DMA_UNIT_INT_NUM = 16;
constexpr int64_t UB_SINGLE_DMA_SIZE_MAX = 95 * 1024;

extern "C" __attribute__((always_inline)) inline __aicore__ void all_gather_origin(
    GM_ADDR input, GM_ADDR output, GM_ADDR gva,
    int64_t max_gva_num, int elements, int copy_len, int64_t magic)
{
    const int64_t aivNum = GetBlockNum() * 2;
    const int64_t aivIndex = GetBlockIdx();

    const int64_t dataOffsetNum = aivNum * MEM_DMA_UNIT_INT_NUM;
    const int64_t flagOffset1st = MEM_DMA_UNIT_INT_NUM * aivIndex;

    int64_t my_rank = shmem_my_pe();
    int64_t pe_size = shmem_n_pes();

    __gm__ int32_t* input_gm = (__gm__ int32_t *)input;
    __gm__ int32_t* output_gm = (__gm__ int32_t *)output;
    __gm__ int32_t* gva_gm = (__gm__ int32_t *)gva;

    // 0-7 copy data to local symmetric mem, 8-15 copy remote data from symmetric mem.
    int corePerGroup = aivNum / 2;
    int corePerRank = corePerGroup / pe_size;
    int lenPerCore = copy_len / corePerGroup;

    // signal_op needed
    __ubuf__ int32_t* ctrlFlagsUB1[16];
    __ubuf__ int32_t* ctrlFlagsUB2[16];
    for (int i = 0; i * 8 < 128; i++) {
        ctrlFlagsUB1[i] = (__ubuf__ int32_t*)(32) + i * 16;
        ctrlFlagsUB2[i] = (__ubuf__ int32_t*)(544) + i * 16;
    }

    int64_t case_offset = magic / 1024 / 1024 * 1024;

    // GM to SymmPtr
    if (aivIndex < corePerGroup) {
        __ubuf__ int32_t* tmpBuff = reinterpret_cast<__ubuf__ int32_t*>(uint64_t(1024 + 32));
        uint32_t copy_ub_size = 190 * 1024;
        uint32_t copy_ub_num = copy_ub_size / sizeof(int32_t);
        uint32_t copy_total_size = lenPerCore * sizeof(int32_t);

        int64_t times = 0;
        int64_t flag = 0;
        while (copy_total_size >= copy_ub_size) {
            shmem_mte_put_mem_nbi(
                gva_gm + max_gva_num + aivIndex * lenPerCore + times * copy_ub_num,
                input_gm + aivIndex * lenPerCore + times * copy_ub_num,
                tmpBuff, copy_ub_size, copy_ub_num, my_rank, EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            times += 1;
            flag = times + magic;
            shmemx_signal_op(gva_gm + case_offset + flagOffset1st, flag, SHMEM_SIGNAL_SET, my_rank);

            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

            copy_total_size -= copy_ub_size;
        }
        if (copy_total_size <= 0) {
            return;
        }
        shmem_mte_put_mem_nbi(
            gva_gm + max_gva_num + aivIndex * lenPerCore + times * copy_ub_num,
            input_gm + aivIndex * lenPerCore + times * copy_ub_num,
            tmpBuff, copy_ub_size, copy_total_size / sizeof(int32_t), my_rank, EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        times += 1;
        flag = times + magic;
        AscendC::PipeBarrier<PIPE_ALL>();
        shmemx_signal_op(gva_gm + case_offset + flagOffset1st, flag, SHMEM_SIGNAL_SET, my_rank);
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }

    // while style
    for (int64_t i = 0; i < corePerGroup; i++) {
        *ctrlFlagsUB1[i] = 0;
        *ctrlFlagsUB2[i] = 0;
    }

    __ubuf__ int32_t* syncBuff = reinterpret_cast<__ubuf__ int32_t*>(uint64_t(0));
    __ubuf__ int32_t* tmpBuff1 = reinterpret_cast<__ubuf__ int32_t*>(uint64_t(1 * 1024 + 32));
    __ubuf__ int32_t* tmpBuff2 = reinterpret_cast<__ubuf__ int32_t*>(uint64_t(96 * 1024 + 32));
    uint32_t copy_ub_size = UB_SINGLE_DMA_SIZE_MAX;
    uint32_t copy_ub_num = copy_ub_size / sizeof(int32_t);
    int x = (aivIndex - corePerGroup) / corePerRank;

    int pingpongId = 0;
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    while (true) {
        for (int blockGroupIdx = 0; blockGroupIdx < corePerGroup; blockGroupIdx++) {
            if (*ctrlFlagsUB1[blockGroupIdx] == INT32_MAX) {
                continue;
            }

            int64_t allDataSizeNeed = lenPerCore * sizeof(int32_t);
            if (*ctrlFlagsUB1[blockGroupIdx] * 190 * 1024 >= allDataSizeNeed) {
                *ctrlFlagsUB1[blockGroupIdx] = INT32_MAX;
                continue;
            }

            shmem_get_int32_mem_nbi(ctrlFlagsUB2[blockGroupIdx], gva_gm + case_offset + blockGroupIdx * MEM_DMA_UNIT_INT_NUM, 1, x);
            AscendC::PipeBarrier<PIPE_ALL>();

            if ((*ctrlFlagsUB2[blockGroupIdx] >> 10) != (magic >> 10)) {
                continue;
            }

            int64_t readyCount = *ctrlFlagsUB2[blockGroupIdx] - magic;
            if (readyCount <= 0 || *ctrlFlagsUB1[blockGroupIdx] >= readyCount) {
                continue;
            }

            int group_recv_offset = x * elements + blockGroupIdx * lenPerCore;
            int group_send_offset = blockGroupIdx * lenPerCore;

            int send_offset = *ctrlFlagsUB1[blockGroupIdx] * 190 * 1024 / sizeof(int32_t);
            int recv_offset = *ctrlFlagsUB1[blockGroupIdx] * 190 * 1024 / sizeof(int32_t);
            int numTotal = (readyCount - *ctrlFlagsUB1[blockGroupIdx]) * 190 * 1024 / sizeof(int32_t);
            if (readyCount * 190 * 1024 > allDataSizeNeed) {
                numTotal = (allDataSizeNeed - *ctrlFlagsUB1[blockGroupIdx] * 190 * 1024) / sizeof(int32_t);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int i = 0; numTotal > 0; i++) {
                AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
                __ubuf__ int32_t* buf = pingpongId == 0 ? tmpBuff1 : tmpBuff2;

                uint32_t copy_num = numTotal > copy_ub_num ? copy_ub_num : numTotal;

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
                shmem_mte_get_mem_nbi(output_gm + group_recv_offset + recv_offset, gva_gm + max_gva_num + group_send_offset + send_offset, buf, copy_ub_size, copy_num, x, EVENT_ID);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);

                send_offset += copy_num;
                recv_offset += copy_num;
                numTotal -= copy_num;
                pingpongId = 1 - pingpongId;
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            *ctrlFlagsUB1[blockGroupIdx] = readyCount;
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        bool finished = true;
        for (int64_t blockGroupIdx = 0; blockGroupIdx < corePerGroup; blockGroupIdx++) {
            if (*ctrlFlagsUB1[blockGroupIdx] != INT32_MAX) {
                finished = false;
                break;
            }
        }
        if (finished) {
            break;
        }
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// all_gather
extern "C" __global__ __aicore__ void all_gather_big_data(GM_ADDR fftsAddr, GM_ADDR input, GM_ADDR output, GM_ADDR gva, int elements, int magic)
{
#ifdef __DAV_C220_VEC__
    AscendC::SetSyncBaseAddr(reinterpret_cast<uint64_t>(fftsAddr));

    const int64_t dataOffsetNum = GetBlockNum() * 2 * MEM_DMA_UNIT_INT_NUM;
    const int64_t flagOffset1st = MEM_DMA_UNIT_INT_NUM * (GetBlockIdx() / 2);

    int64_t my_rank = shmem_my_pe();
    int64_t pe_size = shmem_n_pes();

    const int64_t max_gva_memory = 100 * 1024 * 1024; // Byte
    const int64_t max_gva_num = max_gva_memory / sizeof(int32_t);
    int times = (elements + max_gva_num - 1) / max_gva_num;
    int totalNum = elements;

    __ubuf__ int64_t* ctrlFlagsUB = (__ubuf__ int64_t*)(0);
    for (int i = 0; i < times; i++) {
        *ctrlFlagsUB = 0;
        AscendC::PipeBarrier<PIPE_ALL>();

        int32_t copy_len = totalNum > max_gva_num ? max_gva_num : totalNum;
        shmemx_barrier_all_vec();
        all_gather_origin(
            input + i * max_gva_num * sizeof(int32_t),
            output + i * max_gva_num * sizeof(int32_t),
            gva,
            max_gva_num, elements, copy_len, (magic + i) * 1024);
        totalNum -= max_gva_num;
        AscendC::PipeBarrier<PIPE_ALL>();
    }
#endif
}

// all_gather
extern "C" __global__ __aicore__ void all_gather_small_data(GM_ADDR fftsAddr, GM_ADDR input, GM_ADDR output, GM_ADDR gva, int elements, int magic)
{
#ifdef __DAV_C220_VEC__
    const int64_t aivNum = GetBlockNum() * 2;
    const int64_t aivIndex = GetBlockIdx();

    const int64_t dataOffsetNum = aivNum * MEM_DMA_UNIT_INT_NUM;
    const int64_t flagOffset1st = MEM_DMA_UNIT_INT_NUM * aivIndex;
    const int64_t flagOffset2nd = aivNum * MEM_DMA_UNIT_INT_NUM + flagOffset1st;
    int64_t my_rank = shmem_my_pe();
    int64_t pe_size = shmem_n_pes();
    int64_t max_gva_num = 100 * 1024 * 1024 / sizeof(int32_t);
    int times = (elements + max_gva_num - 1) / max_gva_num;
    int totalNum = elements;

    __ubuf__ int64_t* ctrlFlagsUB = (__ubuf__ int64_t*)(0);
    __ubuf__ int32_t* inputUB[2] = {(__ubuf__ int32_t*)(64), (__ubuf__ int32_t*)(97312)};
    __gm__ int32_t *input_gm = (__gm__ int32_t *)input;
    __gm__ int32_t *output_gm = (__gm__ int32_t *)output;
    __gm__ int32_t *gva_gm = (__gm__ int32_t *)gva;

    // data move parameters
    const uint32_t ub_size = 190 * 1024;
    uint32_t input_offset, output_offset, gva_offset, dataNumRemain;

    // [AllGather Step 1] local input gm -> symmetric mem.
    dataNumRemain = elements / aivNum;
    input_offset = aivIndex * dataNumRemain;
    gva_offset = aivIndex * dataNumRemain;
    if (aivIndex == aivNum - 1) {
        dataNumRemain = elements - dataNumRemain * aivIndex;
    }
    shmem_mte_put_mem_nbi(gva_gm + gva_offset + dataOffsetNum, input_gm + input_offset, inputUB[0], ub_size, dataNumRemain, my_rank, EVENT_ID0);

    const int64_t corePerRank = aivNum / pe_size;
    const int64_t coreSegmentedIdx = aivIndex % corePerRank;
    const int64_t x = aivIndex / corePerRank;

    // Sync Ensure Corresponding Tasks Done.
    shmemx_signal_op((__gm__ int32_t*)gva_gm + flagOffset1st, magic, SHMEM_SIGNAL_SET, my_rank);

    for (int64_t i = 0; i < aivNum; i++) {
        shmem_signal_wait_until((__gm__ int32_t *)shmem_ptr(gva_gm, x) + flagOffset1st, SHMEM_CMP_EQ, magic);
    }

    // [AllGather Step 2] symmetric mem -> local output.
    dataNumRemain = elements / corePerRank;
    output_offset = x * elements + coreSegmentedIdx * dataNumRemain;
    gva_offset = coreSegmentedIdx * dataNumRemain;
    if (coreSegmentedIdx == corePerRank - 1) {
        dataNumRemain = elements - dataNumRemain * coreSegmentedIdx;
    }
    shmem_mte_get_mem_nbi(output_gm + output_offset, gva_gm + gva_offset + dataOffsetNum, inputUB[0], ub_size, dataNumRemain, x, EVENT_ID0);
#endif
}

void allgather_demo(uint32_t block_dim, void* stream, uint8_t *fftsAddr, uint8_t* input, uint8_t* output, uint8_t* gva, int elements, int magic)
{
    if (elements < 2097152) {
        all_gather_small_data<<<block_dim, nullptr, stream>>>(fftsAddr, input, output, gva, elements, magic);
    }
    else {
        all_gather_big_data<<<block_dim, nullptr, stream>>>(fftsAddr, input, output, gva, elements, magic);
    }
}

extern "C" {
uint32_t GetAscendCoreSyncAddr(void **addr);
}

int test_shmem_team_all_gather(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    shmem_init_attr_t *attributes;
    status = shmem_set_attr(rank_id, n_ranks, local_mem_size, ipport, &attributes);
    status = shmem_init_attr(attributes);

    // Prepare FFTS address
    uint8_t *fftsAddr{ nullptr };
    GetAscendCoreSyncAddr(reinterpret_cast<void **>(&fftsAddr));

    int PERF_TIMES = 50;

    int case_num = 24;
    std::vector<uint32_t> test_cases = {};
    for (int i = 0; i < case_num; i++) {
        int data_len = 16 * (1 << i);
        test_cases.push_back(data_len);
    }

    uint32_t BLOCK_NUM = 8;

    std::ofstream outFile("./results.csv");
    if (!outFile.is_open()) {
        std::cerr << "错误：无法创建文件！" << std::endl;
        return 1;
    }
    outFile << "M,N,Time(us)\n";

    // magic used to sync.
    int magic = 1;

    for (int i = 0; i < test_cases.size(); i++) {
        if (rank_id == 0) {
            std::cout << "Case: " << test_cases[i] << " Started." << std::endl;
        }
        uint32_t trans_size = test_cases[i];
        if (trans_size < 2097152) {
            BLOCK_NUM = 4;
        } else {
            BLOCK_NUM = 8;
        }

        void *input_ptr;
        aclrtMalloc(&input_ptr, trans_size * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);

        void *output_ptr;
        aclrtMalloc(&output_ptr, trans_size * n_ranks * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);

        // data Buffer + sync Buffer
        void *ptr = shmem_malloc(128 * 1024 * 1024 * sizeof(int32_t));
        std::vector<int> sync_array(2048, 0);
        aclrtMemcpy((uint8_t*)ptr, 2048 * sizeof(int32_t), sync_array.data(), 2048 * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

        // Input Host Data Init
        std::vector<int32_t> input(trans_size, 0);
        for (int zz = 0; zz < trans_size; zz++) {
            input[zz] = (rank_id * trans_size + zz);
        }

        status = aclrtMemcpy(input_ptr, trans_size * sizeof(int32_t), input.data(), trans_size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

        // AllGather
        for (int zz = 0; zz < PERF_TIMES; zz++) {
            magic++;
            allgather_demo(BLOCK_NUM, stream, (uint8_t *)fftsAddr, (uint8_t *)input_ptr, (uint8_t *)output_ptr, (uint8_t *)ptr, trans_size, magic * 1024);
        }
        status = aclrtSynchronizeStream(stream);

        // Result Check
        int32_t *y_host;
        size_t output_size = n_ranks * trans_size * sizeof(int32_t);
        status = aclrtMallocHost(reinterpret_cast<void**>(&y_host), output_size);
        status = aclrtMemcpy(y_host, output_size, output_ptr, output_size, ACL_MEMCPY_DEVICE_TO_HOST);
        for (int zz = 0; zz < n_ranks; zz++) {
            for (int j = 0; j < trans_size; j++) {
                if (y_host[zz * trans_size + j] != zz * trans_size + j) {
                    std::cout << y_host[zz * trans_size + j] << " != " << zz * trans_size + j << ", trans_size is : " << trans_size << ", idx is: " << zz * trans_size + j << ", rank_id is: "<< rank_id << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
        }

        status = aclrtFreeHost(y_host);
        shmem_free(ptr);
        aclrtFree(input_ptr);
        aclrtFree(output_ptr);

        outFile << 1 << "," << trans_size << "," << " " << "\n";

        if (rank_id == 0) {
            std::cout << "Case: " << test_cases[i] << " Finised !!" << std::endl;
        }
    }

    outFile.close();

    status = shmem_finalize();
    status = aclrtDestroyStream(stream);
    status = aclrtResetDevice(device_id);
    status = aclFinalize();
    return 0;
}

int main(int argc, char *argv[])
{
    int status = 0;
    int n_ranks = atoi(argv[1]);
    int rank_id = atoi(argv[2]);
    ipport = argv[3];
    g_npus = atoi(argv[4]);
    f_rank = atoi(argv[5]);
    f_npu = atoi(argv[6]);
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    status = test_shmem_team_all_gather(rank_id, n_ranks, local_mem_size);
    if (status) {
        std::exit(EXIT_FAILURE);
    }
    
    std::cout << "[SUCCESS] demo run success in rank " << rank_id << std::endl;
    
    return 0;
}
