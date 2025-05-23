#include <iostream>
#include <unistd.h>
#include <acl/acl.h>
#include "shmem_api.h"
#include "shmemi_host_common.h"
#include <gtest/gtest.h>
extern int testGNpuNum;
extern const char* testGlobalIpport;
extern int testFirstNpu;
extern void TestMutilTask(std::function<void(int, int, uint64_t)> func, uint64_t localMemSize, int processCount);

namespace shm {
extern shmem_init_attr_t gAttr;
}

void TestShmemInit(int rankId, int nRanks, uint64_t localMemSize) {
    uint32_t deviceId = rankId % testGNpuNum + testFirstNpu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(deviceId), 0);
    shmem_init_attr_t* attributes;
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
    status = shmem_finalize();
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(deviceId), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void TestShmemInitAttrT(int rankId, int nRanks, uint64_t localMemSize) {
    uint32_t deviceId = rankId % testGNpuNum + testFirstNpu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(deviceId), 0);

    shmem_init_attr_t* attributes = new shmem_init_attr_t{rankId, nRanks, testGlobalIpport, localMemSize, {0, SHMEM_DATA_OP_MTE, 120, 120, 120}};
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
    status = shmem_finalize();
    delete attributes;
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(deviceId), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void TestShmemInitInvalidRankId(int rankId, int nRanks, uint64_t localMemSize) {
    int erankId = -1;
    uint32_t deviceId = rankId % testGNpuNum + testFirstNpu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(deviceId), 0);
    shmem_init_attr_t* attributes;
    shmem_set_attr(erankId, nRanks, localMemSize, testGlobalIpport, &attributes);
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, SHMEM_INVALID_VALUE);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITALIZED);
    EXPECT_EQ(aclrtResetDevice(deviceId), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void TestShmemInitRankIdOverSize(int rankId, int nRanks, uint64_t localMemSize) {
    uint32_t deviceId = rankId % testGNpuNum + testFirstNpu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(deviceId), 0);
    shmem_init_attr_t* attributes;
    shmem_set_attr(rankId + nRanks, nRanks, localMemSize, testGlobalIpport, &attributes);
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, SHMEM_INVALID_PARAM);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITALIZED);
    EXPECT_EQ(aclrtResetDevice(deviceId), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void TestShmemInitZeroMem(int rankId, int nRanks, uint64_t localMemSize) {
    //localMemSize = 0
    uint32_t deviceId = rankId % testGNpuNum + testFirstNpu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(deviceId), 0);
    shmem_init_attr_t* attributes;
    shmem_set_attr(rankId, nRanks, localMemSize, testGlobalIpport, &attributes);
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, SHMEM_INVALID_VALUE);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITALIZED);
    EXPECT_EQ(aclrtResetDevice(deviceId), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void TestShmemInitInvalidMem(int rankId, int nRanks, uint64_t localMemSize) {
    //localMemSize = invalid
    uint32_t deviceId = rankId % testGNpuNum + testFirstNpu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(deviceId), 0);
    shmem_init_attr_t* attributes;
    shmem_set_attr(rankId, nRanks, localMemSize, testGlobalIpport, &attributes);
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, SHMEM_SMEM_ERROR);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITALIZED);
    EXPECT_EQ(aclrtResetDevice(deviceId), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void TestShmemSetConfig(int rankId, int nRanks, uint64_t localMemSize) {
    uint32_t deviceId = rankId % testGNpuNum + testFirstNpu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(deviceId), 0);
    shmem_init_attr_t* attributes;
    shmem_set_attr(rankId, nRanks, localMemSize, testGlobalIpport, &attributes);

    shmem_set_data_op_engine_type(attributes, SHMEM_DATA_OP_MTE);
    shmem_set_timeout(attributes, 50);
    EXPECT_EQ(shm::gAttr.optionAttr.controlOperationTimeout, 50);
    EXPECT_EQ(shm::gAttr.optionAttr.dataOpEngineType, SHMEM_DATA_OP_MTE);
    
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(shm::gState.mype, rankId);
    EXPECT_EQ(shm::gState.npes, nRanks);
    EXPECT_NE(shm::gState.heapBase, nullptr);
    EXPECT_NE(shm::gState.p2pHeapBase[rankId], nullptr);
    EXPECT_EQ(shm::gState.heapSize, localMemSize + SHMEM_EXTRA_SIZE);
    EXPECT_NE(shm::gState.teamPools[0], nullptr);

    EXPECT_EQ(shm::gAttr.optionAttr.controlOperationTimeout, 50);
    EXPECT_EQ(shm::gAttr.optionAttr.dataOpEngineType, SHMEM_DATA_OP_MTE);

    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_IS_INITALIZED);
    status = shmem_finalize();
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(deviceId), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

TEST(TestInitAPI, TestShmemInit)
{   
    const int processCount = testGNpuNum;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemInit, localMemSize, processCount);
}

TEST(TestInitAPI, TestShmemInitAttrT)
{   
    const int processCount = testGNpuNum;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemInitAttrT, localMemSize, processCount);
}

TEST(TestInitAPI, TestShmemInitErrorInvalidRankId)
{   
    const int processCount = testGNpuNum;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemInitInvalidRankId, localMemSize, processCount);
}

TEST(TestInitAPI, TestShmemInitErrorRankIdOversize)
{   
    const int processCount = testGNpuNum;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemInitRankIdOverSize, localMemSize, processCount);
}

TEST(TestInitAPI, TestShmemInitErrorZeroMem)
{   
    const int processCount = testGNpuNum;
    uint64_t localMemSize = 0;
    TestMutilTask(TestShmemInitZeroMem, localMemSize, processCount);
}

TEST(TestInitAPI, TestShmemInitErrorInvalidMem)
{   
    const int processCount = testGNpuNum;
    uint64_t localMemSize = 1024UL * 1024UL;
    TestMutilTask(TestShmemInitInvalidMem, localMemSize, processCount);
}

TEST(TestInitAPI, TestSetConfig)
{   
    const int processCount = testGNpuNum;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemSetConfig, localMemSize, processCount);
}