/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 */
#include <cstdint>
#include <unordered_set>
#include <gtest/gtest.h>

#include "acl/acl.h"

#include "host/shmem_host_init.h"
#include "host/shmem_host_heap.h"
#include "shmemi_init.h"
#include "shmemi_mm.h"

extern int testGNpuNum;
extern int testFirstNpu;
extern const char *testGlobalIpport;
extern void TestMutilTask(std::function<void(int, int, uint64_t)> func, uint64_t localMemSize, int processCount);
extern void TestInit(int rankId, int nRanks, uint64_t localMemSize, aclrtStream *st);
extern void TestFinalize(aclrtStream stream, int deviceId);

static uint8_t *const HeapMemoryStart = (uint8_t *)(ptrdiff_t)0x100000000UL;
static uint64_t HeapMemorySize = 4UL * 1024UL * 1024UL;
static aclrtStream HeapMemoryStream;

class ShareMemoryManagerTest : public testing::Test {
public:
    void TearDown() override
    {
        Finalize();
    }

protected:
    void Initialize(int rankId, int nRanks, uint64_t localMemSize)
    {
        uint32_t deviceId = rankId % testGNpuNum + testFirstNpu;
        int status = SHMEM_SUCCESS;
        EXPECT_EQ(aclInit(nullptr), 0);
        EXPECT_EQ(status = aclrtSetDevice(deviceId), 0);
        shmem_init_attr_t *attributes;
        shmem_set_attr(rankId, nRanks, localMemSize, testGlobalIpport, &attributes);
        status = shmem_init_attr(attributes);
        EXPECT_EQ(status, SHMEM_SUCCESS);
        EXPECT_EQ(shm::gState.mype, rankId);
        EXPECT_EQ(shm::gState.npes, nRanks);
        EXPECT_NE(shm::gState.heapBase, nullptr);
        EXPECT_NE(shm::gState.p2pHeapBase[rankId], nullptr);
        EXPECT_EQ(shm::gState.heapSize, localMemSize + SHMEM_EXTRA_SIZE);
        EXPECT_NE(shm::gState.teamPools[0], nullptr);
        status = shmem_init_status();
        EXPECT_EQ(status, SHMEM_STATUS_IS_INITALIZED);
        testingRank = true;
    }

    void Finalize()
    {
        if (testingRank) {
            auto status = shmem_finalize();
            EXPECT_EQ(status, SHMEM_SUCCESS);
            testingRank = false;
        }
    }

    bool testingRank = false;
};

TEST_F(ShareMemoryManagerTest, allocate_one_piece_success)
{
    const int processCount = testGNpuNum;
    uint64_t localMemSize = HeapMemorySize;
    TestMutilTask(
        [this](int rankId, int nRanks, uint64_t localMemSize) {
            Initialize(rankId, nRanks, localMemSize);
            auto ptr = shmem_malloc(4096UL);
            EXPECT_NE(nullptr, ptr);
        },
        localMemSize, processCount);
}

TEST_F(ShareMemoryManagerTest, allocate_full_space_success)
{
    const int processCount = testGNpuNum;
    uint64_t localMemSize = HeapMemorySize;
    TestMutilTask(
        [this](int rankId, int nRanks, uint64_t localMemSize) {
            Initialize(rankId, nRanks, localMemSize);
            auto ptr = shmem_malloc(HeapMemorySize);
            EXPECT_NE(nullptr, ptr);
        },
        localMemSize, processCount);
}

TEST_F(ShareMemoryManagerTest, allocate_larage_memory_failed)
{
    const int processCount = testGNpuNum;
    uint64_t localMemSize = HeapMemorySize;
    TestMutilTask(
        [this](int rankId, int nRanks, uint64_t localMemSize) {
            Initialize(rankId, nRanks, localMemSize);
            auto ptr = shmem_malloc(HeapMemorySize + 1UL);
            EXPECT_EQ(nullptr, ptr);
        },
        localMemSize, processCount);
}

TEST_F(ShareMemoryManagerTest, free_merge)
{
    const int processCount = testGNpuNum;
    uint64_t localMemSize = HeapMemorySize;
    TestMutilTask(
        [this](int rankId, int nRanks, uint64_t localMemSize) {
            Initialize(rankId, nRanks, localMemSize);
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
        },
        localMemSize, processCount);
}
