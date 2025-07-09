/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "fp16_t.h"
#include "bfloat16.h"
#include "../matmul_allreduce/utils/utils.h"

using fp16_t = op::fp16_t;
using bfloat16 = op::bfloat16;

#include "acl/acl.h"
#include "shmem_api.h"

int g_npus = 8;
const char *ipport;
int f_rank = 0;
int f_npu = 0;
const char *data_type;

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;

template<class T>
extern void allgather_demo(uint32_t block_dim, void* stream, uint8_t *fftsAddr, uint8_t* input, uint8_t* output, uint8_t* gva, int elements, int magic);

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

    // magic is used to sync.
    int magic = 1;

    for (int i = 0; i < test_cases.size(); i++) {
        if (rank_id == 0) {
            std::cout << "Case: " << test_cases[i] << " Started." << std::endl;
        }
        uint32_t trans_size = test_cases[i];

        //  Small data kernel needs 8 AIV core, Big data kernel needs 16 AIV.
        if (trans_size * sizeof(T) < 2097152) {
            BLOCK_NUM = 4;
        } else {
            BLOCK_NUM = 8;
        }

        void *input_ptr;
        aclrtMalloc(&input_ptr, trans_size * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
        uint8_t *input_host;
        aclrtMallocHost((void **)(&input_host), trans_size * sizeof(T));
        std::string inputFile = "../../examples/allgather/golden/allgather_" + std::to_string(trans_size) + "_" + std::to_string(n_ranks) + "/input_gm_" + std::to_string(rank_id) + ".bin";
        ReadFile(inputFile, input_host, trans_size * sizeof(T));
        aclrtMemcpy(input_ptr, trans_size * sizeof(T), input_host, trans_size * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

        void *output_ptr;
        aclrtMalloc(&output_ptr, trans_size * n_ranks * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);

        // sync Buffer + data Buffer
        int aiv_num = BLOCK_NUM * 2;
        void *ptr = shmem_malloc(aiv_num * SYNC_FLAG_INTERVAL * sizeof(T) + GVA_BUFF_MAX_SIZE / sizeof(T));

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
        std::string goldenFile = "../../examples/allgather/golden/allgather_" + std::to_string(trans_size) + "_" + std::to_string(n_ranks) + "/golden.bin";
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
        status = test_shmem_team_all_gather<int32_t>(rank_id, n_ranks, local_mem_size);
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