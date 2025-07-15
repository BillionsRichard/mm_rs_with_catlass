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
#include <cstdint>
#include <gtest/gtest.h>
#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "shmem_api.h"

extern int test_gnpu_num;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int process_count);
extern void test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *stream);
extern void test_finalize(aclrtStream stream, int device_id);
extern void get_device_ptr(uint32_t block_dim, void* stream, uint8_t* gva);

static int32_t test_get_device_ptr(aclrtStream stream, uint8_t *ptr, int rank_id, uint32_t rank_size)
{
    int *y_host;
    size_t input_size = 2 * sizeof(int);
    EXPECT_EQ(aclrtMallocHost((void **)(&y_host), input_size), 0);

    uint32_t block_dim = 1;
    int32_t device_id;
    SHMEM_CHECK_RET(aclrtGetDevice(&device_id));

    get_device_ptr(block_dim, stream, ptr);
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(1);

    EXPECT_EQ(aclrtMemcpy(y_host, input_size, ptr, input_size,
                          ACL_MEMCPY_DEVICE_TO_HOST), 0);

    EXPECT_EQ(y_host[0], 1);
    EXPECT_EQ(y_host[1], 1);

    EXPECT_EQ(aclrtFreeHost(y_host), 0);
    return 0;
}

void test_shmem_ptr(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    int *ptr = static_cast<int*>(shmem_malloc(2 * sizeof(int)));
    ASSERT_NE(ptr, nullptr);

    void *host_self = shmem_ptr(ptr, rank_id);
    ASSERT_NE(host_self, nullptr);
    EXPECT_EQ(host_self, ptr);

    int peer = (rank_id + 1) % n_ranks;
    void *host_remote = shmem_ptr(ptr, peer);
    void *next_remote = shmem_ptr(ptr + 1, peer);
    ASSERT_NE(host_remote, nullptr);
    EXPECT_EQ(static_cast<int *>(next_remote) - static_cast<int *>(host_remote), 1);

    auto status = test_get_device_ptr(stream, (uint8_t*)ptr, rank_id, n_ranks);
    EXPECT_EQ(status, SHMEM_SUCCESS);

    shmem_free(ptr);

    std::cerr << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
    if (::testing::Test::HasFailure()) {
        exit(1);
    }
}

TEST(TestMemApi, TestShmemPtr)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_ptr, local_mem_size, process_count);
}

TEST(TestMemApi, TestShmemMteSetUbParams)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            uint64_t offset = 1;
            uint32_t ub_size = 256;
            uint32_t event_id = 0;

            ASSERT_EQ(shmem_mte_set_ub_params(offset, ub_size, event_id), 0);
            ASSERT_EQ(shm::g_state.mte_config.shmem_ub, offset);
            ASSERT_EQ(shm::g_state.mte_config.ub_size, ub_size);
            ASSERT_EQ(shm::g_state.mte_config.event_id, event_id);
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}