#include <iostream>
#include <cstdlib>
#include <string>

#include "acl/acl.h"
#include "shmem_api.h"
#include "shmemi_host_common.h"

#include <gtest/gtest.h>
using namespace std;
extern int test_gnpu_num;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int process_count);
extern void test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st);
extern void test_finalize(aclrtStream stream, int device_id);

extern void team_allgather(uint32_t block_dim, void* stream, uint8_t* gva, shmem_team_t team_id);

void test_shmem_team_all_gather(int rank_id, int n_ranks, uint64_t local_mem_size) {
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);
    
    shmem_team_t team_odd;
    int start = 1;
    int stride = 2;
    int team_size = n_ranks / 2;
    void *ptr = shmem_malloc(1024);
    if (rank_id & 1) {
        // Team split
        shmem_team_split_strided(SHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);

        // Initialize data
        uint32_t trans_size = 16;
        std::vector<int32_t> input(trans_size, 0);
        for (int i = 0; i < trans_size; i++) {
            input[i] = (rank_id + 10);
        }

        ASSERT_EQ(aclrtMemcpy(ptr + shmem_team_my_pe(team_odd) * trans_size * sizeof(int32_t), trans_size, input.data(), trans_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);

        // Execute AllGather
        team_allgather(1, stream, (uint8_t *)ptr, team_odd);
        EXPECT_EQ(aclrtSynchronizeStream(stream), 0);

        // Check results
        int32_t *y_host;
        size_t input_size = team_size * trans_size * sizeof(int32_t);
        EXPECT_EQ(aclrtMallocHost((void **) (&y_host), input_size), 0);
        EXPECT_EQ(aclrtMemcpy(y_host, input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
        
        for (int i = 0; i < team_size; i++) {
            EXPECT_EQ(y_host[trans_size * i], 11 + i * 2);
        }
        
        EXPECT_EQ(aclrtFreeHost(y_host), 0);
        shmem_team_destroy(team_odd);
    }
    shmem_free(ptr);

    std::cerr << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}



TEST(TestTeamFunc, TestShmemTeam)
{   
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_team_all_gather, local_mem_size, process_count);
}