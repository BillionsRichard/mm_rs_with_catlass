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
#include <unistd.h>
#include <acl/acl.h>
#include <gtest/gtest.h>
#include "shmemi_host_common.h"
#include <gtest/gtest.h>
extern int test_gnpu_num;
extern const char* test_global_ipport;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int process_count);

namespace shm {
extern shmem_init_attr_t g_attr;
}

void test_shmem_init(int rank_id, int n_ranks, uint64_t local_mem_size) {
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_init_attr_t* attributes;
    shmem_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(shm::g_state.mype, rank_id);
    EXPECT_EQ(shm::g_state.npes, n_ranks);
    EXPECT_NE(shm::g_state.heap_base, nullptr);
    EXPECT_NE(shm::g_state.p2p_heap_base[rank_id], nullptr);
    EXPECT_EQ(shm::g_state.heap_size, local_mem_size + SHMEM_EXTRA_SIZE);
    EXPECT_NE(shm::g_state.team_pools[0], nullptr);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_IS_INITIALIZED);
    status = shmem_finalize();
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void test_shmem_init_attr(int rank_id, int n_ranks, uint64_t local_mem_size) {
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);

    shmem_init_attr_t* attributes = new shmem_init_attr_t{rank_id, n_ranks, test_global_ipport, local_mem_size, {0, SHMEM_DATA_OP_MTE, 120, 120, 120}};
    status = shmem_init_attr(attributes);

    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(shm::g_state.mype, rank_id);
    EXPECT_EQ(shm::g_state.npes, n_ranks);
    EXPECT_NE(shm::g_state.heap_base, nullptr);
    EXPECT_NE(shm::g_state.p2p_heap_base[rank_id], nullptr);
    EXPECT_EQ(shm::g_state.heap_size, local_mem_size + SHMEM_EXTRA_SIZE);
    EXPECT_NE(shm::g_state.team_pools[0], nullptr);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_IS_INITIALIZED);
    status = shmem_finalize();
    delete attributes;
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void test_shmem_init_invalid_rank_id(int rank_id, int n_ranks, uint64_t local_mem_size) {
    int erank_id = -1;
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_init_attr_t* attributes;
    shmem_set_attr(erank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, SHMEM_INVALID_VALUE);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void test_shmem_init_rank_id_over_size(int rank_id, int n_ranks, uint64_t local_mem_size) {
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_init_attr_t* attributes;
    shmem_set_attr(rank_id + n_ranks, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, SHMEM_INVALID_PARAM);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void test_shmem_init_zero_mem(int rank_id, int n_ranks, uint64_t local_mem_size) {
    //local_mem_size = 0
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_init_attr_t* attributes;
    shmem_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, SHMEM_INVALID_VALUE);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void test_shmem_init_invalid_mem(int rank_id, int n_ranks, uint64_t local_mem_size) {
    //local_mem_size = invalid
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_init_attr_t* attributes;
    shmem_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, SHMEM_SMEM_ERROR);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void test_shmem_init_set_config(int rank_id, int n_ranks, uint64_t local_mem_size) {
    uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_init_attr_t* attributes;
    shmem_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);

    shmem_set_data_op_engine_type(attributes, SHMEM_DATA_OP_MTE);
    shmem_set_timeout(attributes, 50);
    EXPECT_EQ(shm::g_attr.option_attr.control_operation_timeout, 50);
    EXPECT_EQ(shm::g_attr.option_attr.data_op_engine_type, SHMEM_DATA_OP_MTE);
    
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(shm::g_state.mype, rank_id);
    EXPECT_EQ(shm::g_state.npes, n_ranks);
    EXPECT_NE(shm::g_state.heap_base, nullptr);
    EXPECT_NE(shm::g_state.p2p_heap_base[rank_id], nullptr);
    EXPECT_EQ(shm::g_state.heap_size, local_mem_size + SHMEM_EXTRA_SIZE);
    EXPECT_NE(shm::g_state.team_pools[0], nullptr);

    EXPECT_EQ(shm::g_attr.option_attr.control_operation_timeout, 50);
    EXPECT_EQ(shm::g_attr.option_attr.data_op_engine_type, SHMEM_DATA_OP_MTE);

    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_IS_INITIALIZED);
    status = shmem_finalize();
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

TEST(TestInitAPI, TestShmemInit)
{   
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_init, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitAttr)
{   
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_init_attr, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitErrorInvalidRankId)
{   
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_init_invalid_rank_id, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitErrorRankIdOversize)
{   
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_init_rank_id_over_size, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitErrorZeroMem)
{   
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 0;
    test_mutil_task(test_shmem_init_zero_mem, local_mem_size, process_count);
}

TEST(TestInitAPI, TestShmemInitErrorInvalidMem)
{   
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL;
    test_mutil_task(test_shmem_init_invalid_mem, local_mem_size, process_count);
}

TEST(TestInitAPI, TestSetConfig)
{   
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_init_set_config, local_mem_size, process_count);
}