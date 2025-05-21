#include <iostream>
#include <cstdlib>
#include <string>
using namespace std;

#include "acl/acl.h"
#include "shmem_api.h"
#include "shmemi_host_common.h"

#include <gtest/gtest.h>
extern int testGNpuNum;
extern int testFirstNpu;
extern void TestMutilTask(std::function<void(int, int, uint64_t)> func, uint64_t localMemSize, int processCount);
extern void TestInit(int rankId, int nRanks, uint64_t localMemSize, aclrtStream *st);
extern void TestFinalize(aclrtStream stream, int deviceId);

extern void TeamAllGather(uint32_t blockDim, void* stream, uint8_t* gva, shmem_team_t teamId);

void TestShmemTeamAllGather(int rankId, int nRanks, uint64_t localMemSize) {
    int32_t deviceId = rankId % testGNpuNum + testFirstNpu;
    aclrtStream stream;
    TestInit(rankId, nRanks, localMemSize, &stream);
    ASSERT_NE(stream, nullptr);
    
    shmem_team_t teamOdd;
    int start = 1;
    int stride = 2;
    int teamSize = nRanks / 2;
    void *ptr = shmem_malloc(1024);
    if (rankId & 1) {
        // Team split
        shmem_team_split_strided(SHMEM_TEAM_WORLD, start, stride, teamSize, &teamOdd);

        // Initialize data
        uint32_t transSize = 16;
        std::vector<int32_t> input(transSize, 0);
        for (int i = 0; i < transSize; i++) {
            input[i] = (rankId + 10);
        }

        ASSERT_EQ(aclrtMemcpy(ptr + shmem_team_my_pe(teamOdd) * transSize * sizeof(int32_t), transSize, input.data(), transSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);

        // Execute AllGather
        TeamAllGather(1, stream, (uint8_t *)ptr, teamOdd);
        EXPECT_EQ(aclrtSynchronizeStream(stream), 0);

        // Check results
        int32_t *yHost;
        size_t inputSize = teamSize * transSize * sizeof(int32_t);
        EXPECT_EQ(aclrtMallocHost((void **) (&yHost), inputSize), 0);
        EXPECT_EQ(aclrtMemcpy(yHost, inputSize, ptr, inputSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);
        
        for (int i = 0; i < teamSize; i++) {
            EXPECT_EQ(yHost[transSize * i], 11 + i * 2);
        }
        
        EXPECT_EQ(aclrtFreeHost(yHost), 0);
        shmem_team_destroy(teamOdd);
    }
    shmem_free(ptr);

    std::cerr << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    TestFinalize(stream, deviceId);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}



TEST(TestTeamFunc, TestShmemTeam)
{   
    const int processCount = testGNpuNum;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemTeamAllGather, localMemSize, processCount);
}