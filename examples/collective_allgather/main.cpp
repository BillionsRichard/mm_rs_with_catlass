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
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <iomanip>
#include <sys/file.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "helper.hpp"
#include "golden.hpp"
#include "fp16_t.h"
#include "bfloat16.h"
#include "../fusion_matmul_allreduce/utils/utils.h"

using fp16_t = op::fp16_t;
using bfloat16 = op::bfloat16;

#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem_api.h"

int g_npus = 8;
const char *ipport;
int f_rank = 0;
int f_npu = 0;
const char *data_type;

using namespace AscendC;

constexpr int64_t MEM_DMA_UNIT_INT_NUM = 16;
constexpr int64_t UB_SINGLE_DMA_SIZE_MAX = 190 * 1024;

template<typename T>
SHMEM_DEVICE void all_gather_origin(__gm__ T* input, __gm__ T* output, __gm__ T* gva, int64_t max_gva_num, int elements, int copy_len, int64_t magic)
{
    const int64_t aivNum = GetBlockNum() * 2;
    const int64_t aivIndex = GetBlockIdx();

    const int64_t dataOffsetNum = aivNum * MEM_DMA_UNIT_INT_NUM;
    const int64_t flagOffset1st = MEM_DMA_UNIT_INT_NUM * aivIndex;

    int64_t my_rank = shmem_my_pe();
    int64_t pe_size = shmem_n_pes();

    __gm__ T* input_gm = (__gm__ T *)input;
    __gm__ T* output_gm = (__gm__ T *)output;
    __gm__ T* gva_data_gm = (__gm__ T *)((__gm__ int32_t *)gva + dataOffsetNum);
    __gm__ int32_t* gva_sync_gm = (__gm__ int32_t *)gva;

    // signal_op needed
    __ubuf__ int32_t* ctrlFlagsUB1[16];
    __ubuf__ int32_t* ctrlFlagsUB2[16];
    for (int i = 0; i * 8 < 128; i++) {
        ctrlFlagsUB1[i] = (__ubuf__ int32_t*)(32) + i * 16;
        ctrlFlagsUB2[i] = (__ubuf__ int32_t*)(544) + i * 16;
    }

    // avoid sync signal collision on gm.
    int64_t case_offset = magic / 1024 / 1024 * 1024;

    // 0-7 copy data to local symmetric mem, 8-15 copy remote data from symmetric mem.
    int corePerGroup = aivNum / 2;
    int corePerRank = corePerGroup / pe_size;
    int lenPerCore = copy_len / corePerGroup;

    // GM to SymmPtr
    if (aivIndex < corePerGroup) {
        __ubuf__ T* tmpBuff = reinterpret_cast<__ubuf__ T*>(uint64_t(1024 + 32));
        uint32_t copy_ub_size = UB_SINGLE_DMA_SIZE_MAX;
        uint32_t copy_ub_num = copy_ub_size / sizeof(T);
        uint32_t copy_total_size = lenPerCore * sizeof(T);

        int64_t times = 0;
        int64_t flag = 0;
        while (copy_total_size >= copy_ub_size) {
            shmem_mte_put_mem_nbi(
                gva_data_gm + max_gva_num + aivIndex * lenPerCore + times * copy_ub_num,
                input_gm + aivIndex * lenPerCore + times * copy_ub_num,
                tmpBuff, copy_ub_size, copy_ub_num, my_rank, EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            times += 1;
            flag = times + magic;
            shmemx_signal_op(gva_sync_gm + case_offset + flagOffset1st, flag, SHMEM_SIGNAL_SET, my_rank);

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
            gva_data_gm + max_gva_num + aivIndex * lenPerCore + times * copy_ub_num,
            input_gm + aivIndex * lenPerCore + times * copy_ub_num,
            tmpBuff, copy_ub_size, copy_total_size / sizeof(T), my_rank, EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        times += 1;
        flag = times + magic;
        AscendC::PipeBarrier<PIPE_ALL>();
        shmemx_signal_op(gva_sync_gm + case_offset + flagOffset1st, flag, SHMEM_SIGNAL_SET, my_rank);
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }

    // while style
    for (int64_t i = 0; i < corePerGroup; i++) {
        *ctrlFlagsUB1[i] = 0;
        *ctrlFlagsUB2[i] = 0;
    }

    __ubuf__ T* tmpBuff1 = reinterpret_cast<__ubuf__ T*>(uint64_t(1 * 1024 + 32));
    __ubuf__ T* tmpBuff2 = reinterpret_cast<__ubuf__ T*>(uint64_t(96 * 1024 + 32));
    uint32_t copy_ub_size = UB_SINGLE_DMA_SIZE_MAX / 2;
    uint32_t copy_ub_num = copy_ub_size / sizeof(T);
    int x = (aivIndex - corePerGroup) / corePerRank;

    int pingpongId = 0;
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    while (true) {
        for (int blockGroupIdx = 0; blockGroupIdx < corePerGroup; blockGroupIdx++) {
            if (*ctrlFlagsUB1[blockGroupIdx] == INT32_MAX) {
                continue;
            }

            int64_t allDataSizeNeed = lenPerCore * sizeof(T);
            if (*ctrlFlagsUB1[blockGroupIdx] * 190 * 1024 >= allDataSizeNeed) {
                *ctrlFlagsUB1[blockGroupIdx] = INT32_MAX;
                continue;
            }

            shmem_get_int32_mem_nbi(ctrlFlagsUB2[blockGroupIdx], gva_sync_gm + case_offset + blockGroupIdx * MEM_DMA_UNIT_INT_NUM, 1, x);
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

            int send_offset = *ctrlFlagsUB1[blockGroupIdx] * 190 * 1024 / sizeof(T);
            int recv_offset = *ctrlFlagsUB1[blockGroupIdx] * 190 * 1024 / sizeof(T);
            int numTotal = (readyCount - *ctrlFlagsUB1[blockGroupIdx]) * 190 * 1024 / sizeof(T);
            if (readyCount * 190 * 1024 > allDataSizeNeed) {
                numTotal = (allDataSizeNeed - *ctrlFlagsUB1[blockGroupIdx] * 190 * 1024) / sizeof(T);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int i = 0; numTotal > 0; i++) {
                AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
                __ubuf__ T* buf = pingpongId == 0 ? tmpBuff1 : tmpBuff2;

                uint32_t copy_num = numTotal > copy_ub_num ? copy_ub_num : numTotal;

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
                shmem_mte_get_mem_nbi(output_gm + group_recv_offset + recv_offset, gva_data_gm + max_gva_num + group_send_offset + send_offset, buf, copy_ub_size, copy_num, x, EVENT_ID);
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
template<typename T>
SHMEM_DEVICE void all_gather_big_data(GM_ADDR fftsAddr, __gm__ T* input, __gm__ T* output, __gm__ T* gva, int elements, int magic)
{
#ifdef __DAV_C220_VEC__
    AscendC::SetSyncBaseAddr(reinterpret_cast<uint64_t>(fftsAddr));

    const int64_t max_gva_memory = 100 * 1024 * 1024; // Byte
    const int64_t max_gva_num = max_gva_memory / sizeof(T);
    int times = (elements + max_gva_num - 1) / max_gva_num;
    int totalNum = elements;

    __ubuf__ int64_t* ctrlFlagsUB = (__ubuf__ int64_t*)(0);
    for (int i = 0; i < times; i++) {
        *ctrlFlagsUB = 0;
        AscendC::PipeBarrier<PIPE_ALL>();
        int32_t copy_len = totalNum > max_gva_num ? max_gva_num : totalNum;
        shmemx_barrier_all_vec();
        all_gather_origin(input + i * max_gva_num, output + i * max_gva_num, gva, max_gva_num, elements, copy_len, (magic + i) * 1024);
        totalNum -= max_gva_num;
        AscendC::PipeBarrier<PIPE_ALL>();
    }
#endif
}

// all_gather
template<typename T>
SHMEM_DEVICE void all_gather_small_data(GM_ADDR fftsAddr, __gm__ T* input, __gm__ T* output, __gm__ T* gva, int elements, int magic)
{
#ifdef __DAV_C220_VEC__
    const int64_t aivNum = GetBlockNum() * 2;
    const int64_t aivIndex = GetBlockIdx();

    const int64_t dataOffsetNum = aivNum * MEM_DMA_UNIT_INT_NUM;
    const int64_t flagOffset1st = MEM_DMA_UNIT_INT_NUM * aivIndex;
    const int64_t flagOffset2nd = aivNum * MEM_DMA_UNIT_INT_NUM + flagOffset1st;
    int64_t my_rank = shmem_my_pe();
    int64_t pe_size = shmem_n_pes();
    int64_t max_gva_num = 100 * 1024 * 1024 / sizeof(T);
    int times = (elements + max_gva_num - 1) / max_gva_num;
    int totalNum = elements;

    __ubuf__ int64_t* ctrlFlagsUB = (__ubuf__ int64_t*)(0);
    __ubuf__ T* inputUB[2] = {(__ubuf__ T*)(64), (__ubuf__ T*)(97312)};
    __gm__ T *input_gm = (__gm__ T *)input;
    __gm__ T *output_gm = (__gm__ T *)output;

    __gm__ T *gva_data_gm = (__gm__ T*)((__gm__ int32_t*)gva + dataOffsetNum);
    __gm__ int32_t *gva_sync_gm = (__gm__ int32_t *)gva;

    // data move parameters
    const uint32_t ub_size = UB_SINGLE_DMA_SIZE_MAX;
    uint32_t input_offset, output_offset, gva_offset, dataNumRemain;

    // [AllGather Step 1] local input gm -> symmetric mem.
    dataNumRemain = elements / aivNum;
    input_offset = aivIndex * dataNumRemain;
    gva_offset = aivIndex * dataNumRemain;
    if (aivIndex == aivNum - 1) {
        dataNumRemain = elements - dataNumRemain * aivIndex;
    }
    shmem_mte_put_mem_nbi(gva_data_gm + gva_offset, input_gm + input_offset, inputUB[0], ub_size, dataNumRemain, my_rank, EVENT_ID0);

    const int64_t corePerRank = aivNum / pe_size;
    const int64_t coreSegmentedIdx = aivIndex % corePerRank;
    const int64_t x = aivIndex / corePerRank;

    // Sync Ensure Corresponding Tasks Done.
    shmemx_signal_op(gva_sync_gm + flagOffset1st, magic, SHMEM_SIGNAL_SET, my_rank);

    for (int64_t i = 0; i < aivNum; i++) {
        shmem_signal_wait_until((__gm__ int32_t *)shmem_ptr(gva_sync_gm, x) + flagOffset1st, SHMEM_CMP_EQ, magic);
    }

    // [AllGather Step 2] symmetric mem -> local output.
    dataNumRemain = elements / corePerRank;
    output_offset = x * elements + coreSegmentedIdx * dataNumRemain;
    gva_offset = coreSegmentedIdx * dataNumRemain;
    if (coreSegmentedIdx == corePerRank - 1) {
        dataNumRemain = elements - dataNumRemain * coreSegmentedIdx;
    }
    shmem_mte_get_mem_nbi(output_gm + output_offset, gva_data_gm + gva_offset, inputUB[0], ub_size, dataNumRemain, x, EVENT_ID0);
#endif
}

#define ALLGATHER_FUNC_DEF(type) \
extern "C" __global__ __aicore__ void ShmemAllGather_##type(GM_ADDR fftsAddr, GM_ADDR input, GM_ADDR output, GM_ADDR gva, int elements, int magic) {    \
    if (elements < 2097152) {                                                                                                                           \
        all_gather_small_data<type>(fftsAddr, (__gm__ type*)input, (__gm__ type*)output, (__gm__ type*)gva, elements, magic);                           \
    }                                                                                                                                                   \
    else {                                                                                                                                              \
        all_gather_big_data<type>(fftsAddr, (__gm__ type*)input, (__gm__ type*)output, (__gm__ type*)gva, elements, magic);                             \
    }                                                                                                                                                   \
}

#define TYPE_FUNC(fun) \
    fun(int);fun(int8_t);fun(int16_t);fun(int32_t);fun(int64_t); \
    fun(float);fun(float16_t);fun(bfloat16_t)

TYPE_FUNC(ALLGATHER_FUNC_DEF);

template<class T>
void allgather_demo(uint32_t block_dim, void* stream, uint8_t *fftsAddr, uint8_t* input, uint8_t* output, uint8_t* gva, int elements, int magic)
{
    if (std::is_same<T, fp16_t>::value) {
        ShmemAllGather_float16_t<<<block_dim, nullptr, stream>>>(fftsAddr, input, output, gva, elements, magic);
    }
    else if (std::is_same<T, int>::value) {
        ShmemAllGather_int32_t<<<block_dim, nullptr, stream>>>(fftsAddr, input, output, gva, elements, magic);
    }
    else if (std::is_same<T, bfloat16>::value) {
        ShmemAllGather_bfloat16_t<<<block_dim, nullptr, stream>>>(fftsAddr, input, output, gva, elements, magic);
    }
}

extern "C" {
uint32_t GetAscendCoreSyncAddr(void **addr);
}

template<class T>
int test_shmem_team_all_gather(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    // 初始化ACL和SHMEM
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
        aclrtMalloc(&input_ptr, trans_size * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
        uint8_t *input_host;
        aclrtMallocHost((void **)(&input_host), trans_size * sizeof(T));
        std::string inputFile = "../../examples/collective_allgather/golden/allgather_" + std::to_string(trans_size) + "_" + std::to_string(n_ranks) + "/input_gm_" + std::to_string(rank_id) + ".bin";
        ReadFile(inputFile, input_host, trans_size * sizeof(T));
        aclrtMemcpy(input_ptr, trans_size * sizeof(T), input_host, trans_size * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

        void *output_ptr;
        aclrtMalloc(&output_ptr, trans_size * n_ranks * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);

        // data Buffer + sync Buffer
        void *ptr = shmem_malloc(128 * 1024 * 1024 * sizeof(T));
        std::vector<int> sync_array(2048, 0);
        aclrtMemcpy((uint8_t*)ptr, 2048 * sizeof(T), sync_array.data(), 2048 * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

        // AllGather
        for (int zz = 0; zz < PERF_TIMES; zz++) {
            magic++;
            allgather_demo<T>(BLOCK_NUM, stream, (uint8_t *)fftsAddr, (uint8_t *)input_ptr, (uint8_t *)output_ptr, (uint8_t *)ptr, trans_size, magic * 1024);
        }
        status = aclrtSynchronizeStream(stream);

        // Result Check
        T *output_host;
        size_t output_size = n_ranks * trans_size * sizeof(T);
        status = aclrtMallocHost(reinterpret_cast<void**>(&output_host), output_size);
        status = aclrtMemcpy(output_host, output_size, output_ptr, output_size, ACL_MEMCPY_DEVICE_TO_HOST);

        T *golden_host;
        status = aclrtMallocHost(reinterpret_cast<void**>(&golden_host), output_size);
        std::string goldenFile = "../../examples/collective_allgather/golden/allgather_" + std::to_string(trans_size) + "_" + std::to_string(n_ranks) + "/golden.bin";
        ReadFile(goldenFile, golden_host, n_ranks * trans_size * sizeof(T));
        for (int zz = 0; zz < n_ranks * trans_size; zz++) {
            if (static_cast<float>(output_host[zz]) != static_cast<float>(golden_host[zz])) {
                std::cout << static_cast<float>(output_host[zz]) << " != " << static_cast<float>(golden_host[zz]) << ", trans_size is : " << trans_size << ", idx is: " << zz << ", rank_id is: "<< rank_id << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        // 去初始化
        status = aclrtFreeHost(input_host);
        status = aclrtFreeHost(output_host);
        status = aclrtFreeHost(golden_host);

        shmem_free(ptr);
        aclrtFree(input_ptr);
        aclrtFree(output_ptr);

        outFile << 1 << "," << trans_size << "," << " " << "\n";

        if (rank_id == 0) {
            std::cout << "Case: " << test_cases[i] << " Finised !! Result Correct !!" << std::endl;
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
    data_type = argv[7];
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    if (std::string(data_type) == "int") {
        status = test_shmem_team_all_gather<int>(rank_id, n_ranks, local_mem_size);
    }
    else if (std::string(data_type) == "int32_t") {
        status = test_shmem_team_all_gather<int>(rank_id, n_ranks, local_mem_size);
    }
    else if (std::string(data_type) == "float16_t") {
        status = test_shmem_team_all_gather<fp16_t>(rank_id, n_ranks, local_mem_size);
    }
    else if (std::string(data_type) == "bfloat16_t") {
        status = test_shmem_team_all_gather<bfloat16>(rank_id, n_ranks, local_mem_size);
    }
    if (status) {
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[SUCCESS] demo run success in rank " << rank_id << std::endl;

    return 0;
}