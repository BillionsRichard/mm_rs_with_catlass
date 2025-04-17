#include <iostream>
#include <gtest/gtest.h>
#include <acl/acl.h>
#include "data_utils.h"
#include "shmem_api.h"

TEST(TestInitAPI, TestShmemInit)
{
    int rankId = 0;
    int nRanks = 1;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    uint32_t deviceId = 0;
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
}

TEST(TestInitAPIError, TestShmemInitInvalidRankId)
{
    int rankId = -1;
    int nRanks = 1;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    uint32_t deviceId = 0;
    int status = SHMEM_SUCCESS;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    status = ShmemInit(rankId, nRanks, localMemSize);
    EXPECT_EQ(status, ERROR_INVALID_VALUE);
    status = ShmemInitStatus();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITALIZED);
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
}

TEST(TestInitAPIError, TestShmemInitInvalidRankIdnRanks)
{
    int rankId = 1;
    int nRanks = 1;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    uint32_t deviceId = 0;
    int status = SHMEM_SUCCESS;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    status = ShmemInit(rankId, nRanks, localMemSize);
    EXPECT_EQ(status, ERROR_INVALID_PARAM);
    status = ShmemInitStatus();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITALIZED);
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
}

TEST(TestInitAPIError, TestShmemInitZeroMem)
{
    int rankId = 0;
    int nRanks = 1;
    uint64_t localMemSize = 0;
    uint32_t deviceId = 0;
    int status = SHMEM_SUCCESS;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    status = ShmemInit(rankId, nRanks, localMemSize);
    EXPECT_EQ(status, ERROR_INVALID_VALUE);
    status = ShmemInitStatus();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITALIZED);
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
}

TEST(TestInitAPIError, TestShmemInitInvalidMem)
{
    int rankId = 0;
    int nRanks = 1;
    uint64_t localMemSize = 1024UL * 1024UL;
    uint32_t deviceId = 0;
    int status = SHMEM_SUCCESS;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    status = ShmemInit(rankId, nRanks, localMemSize);
    EXPECT_EQ(status, ERROR_SMEM_ERROR);
    status = ShmemInitStatus();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITALIZED);
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
}