/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 */
#include <cstdint>
#include <unordered_set>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmemi_host_common.h"

extern int test_gnpu_num;
extern int test_first_npu;
extern const char *test_global_ipport;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int process_count);
extern void test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st);
extern void test_finalize(aclrtStream stream, int device_id);

static uint8_t *const heap_memory_start = (uint8_t *)(ptrdiff_t)0x100000000UL;
static uint64_t heap_memory_size = 4UL * 1024UL * 1024UL;
static aclrtStream heap_memory_stream;

class ShareMemoryManagerTest : public testing::Test {

protected:
    void Initialize(int rank_id, int n_ranks, uint64_t local_mem_size)
    {
        uint32_t device_id = rank_id % test_gnpu_num + test_first_npu;
        int status = SHMEM_SUCCESS;
        EXPECT_EQ(aclInit(nullptr), 0);
        EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
        shmem_init_attr_t *attributes;
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
        testingRank = true;
    }

    bool testingRank = false;
};

TEST_F(ShareMemoryManagerTest, allocate_one_piece_success)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            auto ptr = shmem_malloc(4096UL);
            EXPECT_NE(nullptr, ptr);
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, allocate_full_space_success)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            auto ptr = shmem_malloc(heap_memory_size);
            EXPECT_NE(nullptr, ptr);
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, allocate_larage_memory_failed)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            auto ptr = shmem_malloc(heap_memory_size + 1UL);
            EXPECT_EQ(nullptr, ptr);
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}

TEST_F(ShareMemoryManagerTest, free_merge)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = heap_memory_size;
    test_mutil_task(
        [this](int rank_id, int n_ranks, uint64_t local_mem_size) {
            int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
            aclrtStream stream;
            test_init(rank_id, n_ranks, local_mem_size, &stream);
            auto size = 1024UL * 1024UL;  // 1MB

            auto ptr1 = shmem_malloc(size);
            ASSERT_NE(nullptr, ptr1);

            auto ptr2 = shmem_malloc(size);
            ASSERT_NE(nullptr, ptr2);

            auto ptr3 = shmem_malloc(size);
            ASSERT_NE(nullptr, ptr3);

            auto ptr4 = shmem_malloc(size);
            ASSERT_NE(nullptr, ptr4);

            shmem_free(ptr2);
            shmem_free(ptr4);

            auto ptr5 = shmem_malloc(size * 2UL);
            ASSERT_EQ(nullptr, ptr5);

            shmem_free(ptr3);

            auto ptr6 = shmem_malloc(size * 3UL);
            ASSERT_NE(nullptr, ptr6);
            test_finalize(stream, device_id);
        },
        local_mem_size, process_count);
}
