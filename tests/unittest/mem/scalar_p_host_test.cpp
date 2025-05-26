#include <iostream>
#include <string>
using namespace std;

#include "acl/acl.h"
#include "shmem_api.h"

#include <gtest/gtest.h>
extern int test_gnpu_num;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int process_count);
extern void test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st);
extern void test_finalize(aclrtStream stream, int device_id);

extern void put_one_num_do(uint32_t block_dim, void* stream, uint8_t* gva, float val);

static int32_t test_scalar_put_get(aclrtStream stream, uint32_t rank_id, uint32_t rank_size)
{
    float *y_host;
    size_t input_size = 1024 * sizeof(float);
    EXPECT_EQ(aclrtMallocHost((void **)(&y_host), input_size), 0); // size = 1024

    uint32_t block_dim = 1;

    float value = 3.5f + (float)rank_id;
    void *ptr = shmem_malloc(1024);
    put_one_num_do(block_dim, stream, (uint8_t *)ptr, value);
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    EXPECT_EQ(aclrtMemcpy(y_host, 1 * sizeof(float), ptr, 1 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST), 0);

    string p_name = "[Process " + to_string(rank_id) + "] ";
    std::cout << p_name << "-----[PUT]------ " << y_host[0] << " ----" << std::endl;

    // for gtest
    int32_t flag = 0;
    if (y_host[0] != (3.5f + (rank_id + rank_size - 1) % rank_size)) flag = 1;

    EXPECT_EQ(aclrtFreeHost(y_host), 0);
    return flag;
}

void test_shmem_scalar_p(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    int status = test_scalar_put_get(stream, rank_id, n_ranks);
    ASSERT_EQ(status, 0);

    std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

TEST(TestScalarPApi, TestShmemScalarP)
{
    const int process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_scalar_p, local_mem_size, process_count);
}