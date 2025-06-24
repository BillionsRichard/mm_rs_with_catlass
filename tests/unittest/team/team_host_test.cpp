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

extern void get_device_state(uint32_t block_dim, void* stream, uint8_t* gva, shmem_team_t team_id);

static int32_t test_get_device_state(aclrtStream stream, uint8_t *gva, uint32_t rank_id, uint32_t rank_size, shmem_team_t team_id, int stride)
{
    int *y_host;
    size_t input_size = 1024 * sizeof(int);
    EXPECT_EQ(aclrtMallocHost((void **) (&y_host), input_size), 0);      // size = 1024

    uint32_t block_dim = 1;
    void *ptr = shmem_malloc(1024);
    int32_t device_id;
    SHMEM_CHECK_RET(aclrtGetDevice(&device_id));
    get_device_state(block_dim, stream, (uint8_t *) ptr, team_id);
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    EXPECT_EQ(aclrtMemcpy(y_host, 5 * sizeof(int), ptr, 5 * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST), 0);

    if (rank_id & 1) {
        EXPECT_EQ(y_host[0], rank_size);
        EXPECT_EQ(y_host[1], rank_id);
        EXPECT_EQ(y_host[2], rank_id / stride);
        EXPECT_EQ(y_host[3], rank_size / stride);
        EXPECT_EQ(y_host[4], (y_host[3] - 1) * stride + rank_id % stride);
    }

    EXPECT_EQ(aclrtFreeHost(y_host), 0);
    return 0;
}

void test_shmem_team(int rank_id, int n_ranks, uint64_t local_mem_size) {
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);
    // #################### 子通信域切分测试 ############################
    shmem_team_t team_odd;
    int start = 1;
    int stride = 2;
    int team_size = n_ranks / 2;
    shmem_team_split_strided(SHMEM_TEAM_WORLD, start, stride, team_size, &team_odd);

    // #################### host侧取值测试 ##############################
    if (rank_id & 1) {
        ASSERT_EQ(shmem_team_n_pes(team_odd), team_size);
        ASSERT_EQ(shmem_team_my_pe(team_odd), rank_id / stride);
        ASSERT_EQ(shmem_n_pes(), n_ranks);
        ASSERT_EQ(shmem_my_pe(), rank_id);
    }

    // #################### device代码测试 ##############################

    auto status = test_get_device_state(stream, (uint8_t *)shm::g_state.heap_base, rank_id, n_ranks, team_odd, stride);
    EXPECT_EQ(status, SHMEM_SUCCESS);

    // #################### 相关资源释放 ################################
    shmem_team_destroy(team_odd);

    std::cerr << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}



TEST(TestTeamApi, TestShmemTeam)
{   
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_team, local_mem_size, process_count);
}