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
    status = ShmemInitStatus();
    EXPECT_EQ(status, SHMEM_STATUS_IS_INITALIZED);
    status = ShmemFinalize();
    EXPECT_EQ(status, SHMEM_SUCCESS);
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
}