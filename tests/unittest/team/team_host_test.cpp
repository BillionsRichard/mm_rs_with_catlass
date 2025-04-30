#include <iostream>
#include <cstdlib>
#include <string>
using namespace std;

#include "acl/acl.h"
#include "shmem_api.h"
#include "shmemi_host_common.h"

#include <gtest/gtest.h>
extern int testGlobalRanks;
extern int testGNpuNum;
extern void TestMutilTask(std::function<void(int, int, uint64_t)> func, uint64_t localMemSize, int processCount);
extern void TestInit(int rankId, int nRanks, uint64_t localMemSize, aclrtStream *st);
extern void TestFinalize(aclrtStream stream, int deviceId);

extern void GetDeviceState(uint32_t blockDim, void* stream, uint8_t* gva, shmem_team_t teamId);

static int32_t TestGetDeviceState(aclrtStream stream, uint8_t *gva, uint32_t rankId, uint32_t rankSize, shmem_team_t teamId, int stride)
{
    int *yHost;
    size_t inputSize = 1024 * sizeof(int);
    EXPECT_EQ(aclrtMallocHost((void **) (&yHost), inputSize), 0);      // size = 1024

    uint32_t blockDim = 1;
    void *ptr = shmem_malloc(1024);
    int32_t deviceId;
    SHMEM_CHECK_RET(aclrtGetDevice(&deviceId));
    GetDeviceState(blockDim, stream, (uint8_t *) ptr, teamId);
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    EXPECT_EQ(aclrtMemcpy(yHost, 5 * sizeof(int), ptr, 5 * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST), 0);

    if (rankId & 1) {
        EXPECT_EQ(yHost[0], rankSize);
        EXPECT_EQ(yHost[1], rankId);
        EXPECT_EQ(yHost[2], rankId / stride);
        EXPECT_EQ(yHost[3], rankSize / stride);
        EXPECT_EQ(yHost[4], stride + rankId % stride);
    }

    EXPECT_EQ(aclrtFreeHost(yHost), 0);
    return 0;
}

void TestShmemTeam(int rankId, int nRanks, uint64_t localMemSize) {
    int32_t deviceId = rankId % testGNpuNum;
    aclrtStream stream;
    TestInit(rankId, nRanks, localMemSize, &stream);
    ASSERT_NE(stream, nullptr);
    // #################### 子通信域切分测试 ############################
    shmem_team_t team_odd;
    int start = 1;
    int stride = 2;
    int team_size = 4;
    shmem_team_split_strided(SHMEM_TEAM_WORLD, start, stride, team_size, team_odd);

    // #################### host侧取值测试 ##############################
    if (rankId & 1) {
        ASSERT_EQ(shmem_team_n_pes(team_odd), team_size);
        ASSERT_EQ(shmem_team_my_pe(team_odd), rankId / stride);
        ASSERT_EQ(shmem_n_pes(), nRanks);
        ASSERT_EQ(shmem_my_pe(), rankId);
    }

    // #################### device代码测试 ##############################

    auto status = TestGetDeviceState(stream, (uint8_t *)shm::gState.heapBase, rankId, nRanks, team_odd, stride);
    EXPECT_EQ(status, SHMEM_SUCCESS);

    // #################### 相关资源释放 ################################
    shmem_team_destroy(team_odd);

    std::cerr << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    TestFinalize(stream, deviceId);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}



TEST(TestTeamApi, TestShmemTeam)
{   
    const int processCount = testGlobalRanks;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemTeam, localMemSize, processCount);
}