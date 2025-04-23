#include <iostream>
#include <unistd.h>
#include <acl/acl.h>
#include "data_utils.h"
#include "shmem_api.h"

#include <gtest/gtest.h>
extern int testGlobalRanks;
extern int testGNpuNum;
extern const char* testGlobalIpport;
extern void TestMutilTask(std::function<void(int, int, uint64_t)> func, uint64_t localMemSize, int processCount);

void TestShmemInit(int rankId, int nRanks, uint64_t localMemSize) {
    uint32_t deviceId = rankId % testGNpuNum;
    int status = SHMEM_SUCCESS;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    status = ShmemInit(rankId, nRanks, localMemSize);
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(shmemDeviceHostState.mype, rankId);
    EXPECT_EQ(shmemDeviceHostState.npes, nRanks);
    EXPECT_NE(shmemDeviceHostState.heapBase, nullptr);
    EXPECT_NE(shmemDeviceHostState.p2pHeapBase[rankId], nullptr);
    EXPECT_EQ(shmemDeviceHostState.heapSize, localMemSize + DEFAULT_EXTRA_SIZE);
    EXPECT_NE(shmemDeviceHostState.teamPools[0], nullptr);
    status = ShmemInitStatus();
    EXPECT_EQ(status, SHMEM_STATUS_IS_INITALIZED);
    status = ShmemFinalize();
    EXPECT_EQ(status, SHMEM_SUCCESS);
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void TestShmemInitInvalidRankId(int rankId, int nRanks, uint64_t localMemSize) {
    int erankId = -1;
    uint32_t deviceId = rankId % testGNpuNum;
    int status = SHMEM_SUCCESS;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    status = ShmemInit(erankId, nRanks, localMemSize);
    EXPECT_EQ(status, ERROR_INVALID_VALUE);
    status = ShmemInitStatus();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITALIZED);
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void TestShmemInitRankIdOverSize(int rankId, int nRanks, uint64_t localMemSize) {
    uint32_t deviceId = rankId % testGNpuNum;
    int status = SHMEM_SUCCESS;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    status = ShmemInit(rankId + nRanks, nRanks, localMemSize);
    EXPECT_EQ(status, ERROR_INVALID_PARAM);
    status = ShmemInitStatus();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITALIZED);
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void TestShmemInitZeroMem(int rankId, int nRanks, uint64_t localMemSize) {
    //localMemSize = 0
    uint32_t deviceId = rankId % testGNpuNum;
    int status = SHMEM_SUCCESS;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    status = ShmemInit(rankId, nRanks, localMemSize);
    EXPECT_EQ(status, ERROR_INVALID_VALUE);
    status = ShmemInitStatus();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITALIZED);
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void TestShmemInitInvalidMem(int rankId, int nRanks, uint64_t localMemSize) {
    //localMemSize = invalid
    uint32_t deviceId = rankId % testGNpuNum;
    int status = SHMEM_SUCCESS;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    status = ShmemInit(rankId, nRanks, localMemSize);
    EXPECT_EQ(status, ERROR_SMEM_ERROR);
    status = ShmemInitStatus();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITALIZED);
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}




TEST(TestInitAPI, TestShmemInit)
{   
    const int processCount = testGlobalRanks;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemInit, localMemSize, processCount);
}

TEST(TestInitAPI, TestShmemInitErrorInvalidRankId)
{   
    const int processCount = testGlobalRanks;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemInitInvalidRankId, localMemSize, processCount);
}

TEST(TestInitAPI, TestShmemInitErrorRankIdOversize)
{   
    const int processCount = testGlobalRanks;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemInitRankIdOverSize, localMemSize, processCount);
}

TEST(TestInitAPI, TestShmemInitErrorZeroMem)
{   
    const int processCount = testGlobalRanks;
    uint64_t localMemSize = 0;
    TestMutilTask(TestShmemInitZeroMem, localMemSize, processCount);
}

TEST(TestInitAPI, TestShmemInitErrorInvalidMem)
{   
    const int processCount = testGlobalRanks;
    uint64_t localMemSize = 1024UL * 1024UL;
    TestMutilTask(TestShmemInitInvalidMem, localMemSize, processCount);
}