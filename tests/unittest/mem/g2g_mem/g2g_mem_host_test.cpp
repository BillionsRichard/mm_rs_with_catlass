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
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "bfloat16.h"
#include "fp16_t.h"
#include "../utils/func_type.h"

extern int test_gnpu_num;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int process_count);
extern void test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st);
extern void test_finalize(aclrtStream stream, int device_id);

#define TEST_FUNC(NAME, TYPE)                                                                           \
extern void test_g2g_##NAME##_put(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev_ptr);    \
extern void test_g2g_##NAME##_get(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev_ptr)

SHMEM_FUNC_TYPE_HOST(TEST_FUNC);

#define TEST_G2G_PUT_GET(NAME, TYPE)                                                                                    \
static void test_g2g_##NAME##_put_get(aclrtStream stream, uint8_t *gva, uint32_t rank_id, uint32_t rank_size)           \
{                                                                                                                       \
    int total_size = 16 * (int)rank_size;                                                                               \
    size_t input_size = total_size * sizeof(TYPE);                                                                      \
                                                                                                                        \
    std::vector<TYPE> input(total_size, 0);                                                                             \
    for (int i = 0; i < 16; i++) {                                                                                      \
        input[i] = static_cast<TYPE>(rank_id + 10);                                                                     \
    }                                                                                                                   \
                                                                                                                        \
    void *dev_ptr;                                                                                                      \
    ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                                        \
                                                                                                                        \
    ASSERT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);                \
                                                                                                                        \
    uint32_t block_dim = 1;                                                                                             \
    void *ptr = shmem_malloc(1024);                                                                                     \
    test_g2g_##NAME##_put(block_dim, stream, (uint8_t *)ptr, (uint8_t *)dev_ptr);                                       \
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                       \
    sleep(2);                                                                                                           \
                                                                                                                        \
    ASSERT_EQ(aclrtMemcpy(input.data(), input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);                    \
                                                                                                                        \
    std::string p_name = "[Process " + std::to_string(rank_id) + "] ";                                                  \
    std::cout << p_name;                                                                                                \
    for (int i = 0; i < total_size; i++) {                                                                              \
        std::cout << static_cast<float>(input[i]) << " ";                                                               \
    }                                                                                                                   \
    std::cout << std::endl;                                                                                             \
                                                                                                                        \
    test_g2g_##NAME##_get(block_dim, stream, (uint8_t *)ptr, (uint8_t *)dev_ptr);                                       \     
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);                                                                       \
    sleep(2);                                                                                                           \
                                                                                                                        \
    ASSERT_EQ(aclrtMemcpy(input.data(), input_size, dev_ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);                \
                                                                                                                        \
    std::cout << p_name;                                                                                                \
    for (int i = 0; i < total_size; i++) {                                                                              \
        std::cout << static_cast<float>(input[i]) << " ";                                                               \
    }                                                                                                                   \
    std::cout << std::endl;                                                                                             \
    /* for gtest */                                                                                                     \
    int32_t flag = 0;                                                                                                   \
    for (int i = 0; i < total_size; i++){                                                                               \
        int stage = i / 16;                                                                                             \
        if (input[i] != static_cast<TYPE>(stage + 10)) flag = 1;                                                        \
    }                                                                                                                   \
    ASSERT_EQ(flag, 0);                                                                                                 \
}

SHMEM_FUNC_TYPE_HOST(TEST_G2G_PUT_GET);

#define TEST_G2G_SHMEM_MEM(NAME,TYPE)                                                           \
void test_g2g_##NAME##_shmem_mem(int rank_id, int n_ranks, uint64_t local_mem_size) {           \
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                               \
    aclrtStream stream;                                                                         \
    test_init(rank_id, n_ranks, local_mem_size, &stream);                                       \
    ASSERT_NE(stream, nullptr);                                                                 \
                                                                                                \
    test_g2g_##NAME##_put_get(stream, (uint8_t *)shm::g_state.heap_base, rank_id, n_ranks);     \
    std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;                \
    test_finalize(stream, device_id);                                                           \
    if (::testing::Test::HasFailure()){                                                         \
        exit(1);                                                                                \
    }                                                                                           \
}

SHMEM_FUNC_TYPE_HOST(TEST_G2G_SHMEM_MEM);

#define TESTAPI(NAME, TYPE)                                                         \
TEST(TestMemApi, TestG2G##NAME##ShmemMem)                                           \
{                                                                                   \
    const int process_count = test_gnpu_num;                                        \
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;                               \
    test_mutil_task(test_g2g_##NAME##_shmem_mem, local_mem_size, process_count);    \
}

SHMEM_FUNC_TYPE_HOST(TESTAPI);