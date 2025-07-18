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

void fence_order_do(void* stream, uint64_t config, uint8_t *addr, int32_t rank_id, int32_t n_ranks);

static void test_fence_order(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size) {
    aclrtStream stream;
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    uint64_t *dev_ptr = (uint64_t*)shmem_malloc(6 * sizeof(uint64_t));
    ASSERT_EQ(aclrtMemset(dev_ptr, 6 * sizeof(uint64_t), 0, 6 * sizeof(uint64_t)), ACL_ERROR_NONE);

    std::vector<uint64_t> host_buf(6, 0);

    fence_order_do(stream, shmemx_get_ffts_config(), (uint8_t*)dev_ptr, rank_id, n_ranks);

    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    ASSERT_EQ(aclrtMemcpy(host_buf.data(),
                          6 * sizeof(uint64_t),
                          dev_ptr,
                          6 * sizeof(uint64_t),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              0);

    if (rank_id == 1) {
        ASSERT_EQ(host_buf[3], 0xCCu);
        ASSERT_EQ(host_buf[4], 0xBBu);
        ASSERT_EQ(host_buf[5], 0xAAu);
    }

    shmem_free(dev_ptr);

    test_finalize(stream, device_id);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

TEST(TEST_SYNC_API, test_fence_order) {
    const int32_t nranks = test_gnpu_num;
    const uint64_t heap_size = 16 * 1024 * 1024;
    test_mutil_task(test_fence_order, heap_size, nranks);
}