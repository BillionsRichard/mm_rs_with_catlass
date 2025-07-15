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
#include "shmem_api.h"
#include "shmemi_host_common.h"

extern int32_t test_gnpu_num;
extern const char* test_global_ipport;
extern int test_first_npu;

void test_mutil_task(std::function<void(int32_t, int32_t, uint64_t)> func, uint64_t local_mem_size, int32_t process_count);
void test_init(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size, aclrtStream *st);
void test_finalize(aclrtStream stream, int32_t device_id);

void quiet_do(void* stream, uint64_t config, uint8_t *addr, uint8_t *dev, int32_t rank_id, int32_t rank_size);

static void test_quiet(int32_t rank_id, int32_t rank_size, uint64_t local_mem_size) {
    aclrtStream stream;
    test_init(rank_id, rank_size, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    int32_t *addr_dev = (int32_t *)shmem_malloc(sizeof(int32_t) * rank_size);
    ASSERT_EQ(aclrtMemset(addr_dev, sizeof(int32_t) * rank_size, 0, sizeof(int32_t)) * rank_size, 0);

    size_t input_size = (int)rank_size * sizeof(int32_t);

    std::vector<int32_t> input(rank_size, 0);
    for (int32_t i = 0; i < rank_size; i++) {
        input[i] = static_cast<int32_t>(rank_id + 10);
    }     
    void *dev_ptr;
    ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);
    ASSERT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);

    int32_t *addr_host;
    ASSERT_EQ(aclrtMallocHost((void **)&addr_host, sizeof(int32_t) * rank_size), 0);

    quiet_do(stream, shmemx_get_ffts_config(), (uint8_t *)addr_dev, (uint8_t *)dev_ptr, rank_id, rank_size);

    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    ASSERT_EQ(aclrtMemcpy(addr_host, sizeof(int32_t) * rank_size, addr_dev, sizeof(int32_t) * rank_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (int32_t i = 0; i < rank_size; ++i) {
        if (i == 0) {
            ASSERT_EQ(addr_host[i], rank_id + 11);
        } else {
            ASSERT_EQ(addr_host[i], rank_id + 10);
        }
    }

    shmem_free(addr_dev);

    int32_t dev_id = rank_id % test_gnpu_num + test_first_npu;
    test_finalize(stream, dev_id);

    if (::testing::Test::HasFailure()) {
        exit(1);
    }
}

TEST(TEST_SYNC_API, test_quiet) {
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_quiet, local_mem_size, process_count);
}

