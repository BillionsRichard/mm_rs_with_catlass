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
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmem_api.h"
#include "bfloat16.h"
#include "fp16_t.h"
#include "../utils/func_type.h"

extern int test_gnpu_num;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int process_count);
extern void test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st);
extern void test_finalize(aclrtStream stream, int device_id);

#define PUT_ONE_NUM_DO(NAME, TYPE)                                                                      \
extern void put_##NAME##_one_num_do(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev);      \
extern void get_##NAME##_one_num_do(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev) 

SHMEM_FUNC_TYPE(PUT_ONE_NUM_DO);

#define TEST_SCALAR_PUT_GET(NAME, TYPE)                                                                                 \
static int32_t test_##NAME##_scalar_put_get(aclrtStream stream, uint32_t rank_id, uint32_t rank_size)                   \
{                                                                                                                       \
    TYPE *y_host;                                                                                                       \
    size_t input_size = 1024 * sizeof(TYPE);                                                                            \
    EXPECT_EQ(aclrtMallocHost((void **)(&y_host), input_size), 0); /* size = 1024 */                                    \
                                                                                                                        \
    void *dev_ptr;                                                                                                      \
    EXPECT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);                                        \
                                                                                                                        \
    uint32_t block_dim = 1;                                                                                             \
                                                                                                                        \
    TYPE value = static_cast<TYPE>(3.5f) + (TYPE)rank_id;                                                               \
    *dev_ptr = value;                                                                                                   \
    void *ptr = shmem_malloc(1024);                                                                                     \
    put_##NAME##_one_num_do(block_dim, stream, (uint8_t *)ptr, (uint8_t *)dev_ptr);                                     \                               
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);                                                                       \
    sleep(2);                                                                                                           \
                                                                                                                        \
    EXPECT_EQ(aclrtMemcpy(y_host, 1 * sizeof(TYPE), ptr, 1 * sizeof(TYPE), ACL_MEMCPY_DEVICE_TO_HOST), 0);              \
                                                                                                                        \
    std::string p_name = "[Process " + std::to_string(rank_id) + "] ";                                                  \
    std::cout << p_name << "-----[PUT]------ " << static_cast<float>(y_host[0]) << " ----" << std::endl;                \
                                                                                                                        \
    /* for gtest */                                                                                                     \
    int32_t flag = 0;                                                                                                   \
    if (y_host[0] != static_cast<TYPE>(3.5f + (rank_id + rank_size - 1) % rank_size)) flag = 1;                         \
                                                                                                                        \
    get_##NAME##_one_num_do(block_dim, stream, (uint8_t *)ptr, (uint8_t *)dev_ptr);                                     \                               
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);                                                                       \
    sleep(2);                                                                                                           \
                                                                                                                        \
    EXPECT_EQ(aclrtMemcpy(y_host, 1 * sizeof(TYPE), dev_ptr, 1 * sizeof(TYPE), ACL_MEMCPY_DEVICE_TO_HOST), 0);          \
                                                                                                                        \
    p_name = "[Process " + std::to_string(rank_id) + "] ";                                                              \
    std::cout << p_name << "-----[GET]------ " << static_cast<float>(y_host[0]) << " ----" << std::endl;                \
                                                                                                                        \
    /* for gtest */                                                                                                     \
    flag = 0;                                                                                                           \
    if (y_host[0] != static_cast<TYPE>(3.5f + rank_id % rank_size)) flag = 1;                                           \
                                                                                                                        \
    EXPECT_EQ(aclrtFreeHost(y_host), 0);                                                                                \
    return flag;                                                                                                        \
}

SHMEM_FUNC_TYPE(TEST_SCALAR_PUT_GET);

#define TEST_SHMEM_SCALAR(NAME, TYPE)                                                       \
void test_##NAME##_shmem_scalar(int rank_id, int n_ranks, uint64_t local_mem_size)          \
{                                                                                           \
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;                           \
    aclrtStream stream;                                                                     \
    test_init(rank_id, n_ranks, local_mem_size, &stream);                                   \
    ASSERT_NE(stream, nullptr);                                                             \
                                                                                            \
    int status = test_##NAME##_scalar_put_get(stream, rank_id, n_ranks);                    \ 
    ASSERT_EQ(status, 0);                                                                   \
                                                                                            \
    std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;            \
    test_finalize(stream, device_id);                                                       \
    if (::testing::Test::HasFailure()){                                                     \
        exit(1);                                                                            \
    }                                                                                       \
}

SHMEM_FUNC_TYPE(TEST_SHMEM_SCALAR);

#define TEST_API(NAME, TYPE)                                                        \
TEST(TestScalarApi, Test##NAME##ShmemScalar)                                        \
{                                                                                   \
    const int process_count = test_gnpu_num;                                        \
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;                               \
    test_mutil_task(test_##NAME##_shmem_scalar, local_mem_size, process_count);     \
}

SHMEM_FUNC_TYPE(TEST_API);