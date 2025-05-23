#include <iostream>
#include <cstdlib>
#include <string>
using namespace std;

#include "acl/acl.h"
#include "shmem_api.h"
#include "shmemi_host_common.h"

#include <gtest/gtest.h>
extern int test_gnpu_num;
extern int testFirstNpu;
extern void TestMutilTask(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int processCount);
extern void TestInit(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st);
extern void TestFinalize(aclrtStream stream, int device_id);

extern void TeamAllGather(uint32_t block_dim, void* stream, uint8_t* gva, shmem_team_t team_id);

void TestShmemTeamAllGather(int rank_id, int n_ranks, uint64_t local_mem_size) {
    int32_t device_id = rank_id % test_gnpu_num + testFirstNpu;
    aclrtStream stream;
    TestInit(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);
    
    shmem_team_t teamOdd;
    int start = 1;
    int stride = 2;
    int teamSize = n_ranks / 2;
    void *ptr = shmem_malloc(1024);
    if (rank_id & 1) {
        // Team split
        shmem_team_split_strided(SHMEM_TEAM_WORLD, start, stride, teamSize, &teamOdd);

        // Initialize data
        uint32_t transSize = 16;
        std::vector<int32_t> input(transSize, 0);
        for (int i = 0; i < transSize; i++) {
            input[i] = (rank_id + 10);
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

    std::cerr << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    TestFinalize(stream, device_id);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}



TEST(TestTeamFunc, TestShmemTeam)
{   
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemTeamAllGather, local_mem_size, processCount);
}