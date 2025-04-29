#include <iostream>
#include <unistd.h>
#include <acl/acl.h>
#include "shmem_host_api.h"
#include "shmemi_host_intf.h"

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
    ShmemInitAttrT* attributes;
    ShmemSetAttr(rankId, nRanks, localMemSize, testGlobalIpport, &attributes);
    status = ShmemInit();
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(gState.mype, rankId);
    EXPECT_EQ(gState.npes, nRanks);
    EXPECT_NE(gState.heapBase, nullptr);
    EXPECT_NE(gState.p2pHeapBase[rankId], nullptr);
    EXPECT_EQ(gState.heapSize, localMemSize + SHMEM_EXTRA_SIZE);
    EXPECT_NE(gState.teamPools[0], nullptr);
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

void TestShmemInitAttrT(int rankId, int nRanks, uint64_t localMemSize) {
    uint32_t deviceId = rankId % testGNpuNum;
    int status = SHMEM_SUCCESS;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));

    ShmemInitAttrT* attributes = new ShmemInitAttrT{0, rankId, nRanks, testGlobalIpport, localMemSize, {SHMEM_DATA_OP_MTE, 120, 120, 120}};
    status = ShmemInitAttr(attributes);

    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(gState.mype, rankId);
    EXPECT_EQ(gState.npes, nRanks);
    EXPECT_NE(gState.heapBase, nullptr);
    EXPECT_NE(gState.p2pHeapBase[rankId], nullptr);
    EXPECT_EQ(gState.heapSize, localMemSize + SHMEM_EXTRA_SIZE);
    EXPECT_NE(gState.teamPools[0], nullptr);
    status = ShmemInitStatus();
    EXPECT_EQ(status, SHMEM_STATUS_IS_INITALIZED);
    status = ShmemFinalize();
    delete attributes;
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
    ShmemInitAttrT* attributes;
    ShmemSetAttr(erankId, nRanks, localMemSize, testGlobalIpport, &attributes);
    status = ShmemInit();
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
    ShmemInitAttrT* attributes;
    ShmemSetAttr(rankId + nRanks, nRanks, localMemSize, testGlobalIpport, &attributes);
    status = ShmemInit();
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
    ShmemInitAttrT* attributes;
    ShmemSetAttr(rankId, nRanks, localMemSize, testGlobalIpport, &attributes);
    status = ShmemInit();
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
    ShmemInitAttrT* attributes;
    ShmemSetAttr(rankId, nRanks, localMemSize, testGlobalIpport, &attributes);
    status = ShmemInit();
    EXPECT_EQ(status, ERROR_SMEM_ERROR);
    status = ShmemInitStatus();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITALIZED);
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void TestShmemSetConfig(int rankId, int nRanks, uint64_t localMemSize) {
    uint32_t deviceId = rankId % testGNpuNum;
    int status = SHMEM_SUCCESS;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    ShmemInitAttrT* attr;
    ShmemSetAttr(rankId, nRanks, localMemSize, testGlobalIpport, &attr);

    SetDataOpEngineType(attr, SHMEM_DATA_OP_MTE);
    SetTimeout(attr, 50);
    EXPECT_EQ(gAttr.optionAttr.controlOperationTimeout, 50);
    EXPECT_EQ(gAttr.optionAttr.dataOpEngineType, SHMEM_DATA_OP_MTE);
    
    status = ShmemInit();
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(gState.mype, rankId);
    EXPECT_EQ(gState.npes, nRanks);
    EXPECT_NE(gState.heapBase, nullptr);
    EXPECT_NE(gState.p2pHeapBase[rankId], nullptr);
    EXPECT_EQ(gState.heapSize, localMemSize + SHMEM_EXTRA_SIZE);
    EXPECT_NE(gState.teamPools[0], nullptr);

    EXPECT_EQ(gAttr.optionAttr.controlOperationTimeout, 50);
    EXPECT_EQ(gAttr.optionAttr.dataOpEngineType, SHMEM_DATA_OP_MTE);

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

TEST(TestInitAPI, TestShmemInit)
{   
    const int processCount = testGlobalRanks;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemInit, localMemSize, processCount);
}

TEST(TestInitAPI, TestShmemInitAttrT)
{   
    const int processCount = testGlobalRanks;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemInitAttrT, localMemSize, processCount);
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

TEST(TestInitAPI, TestSetConfig)
{   
    const int processCount = testGlobalRanks;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemSetConfig, localMemSize, processCount);
}