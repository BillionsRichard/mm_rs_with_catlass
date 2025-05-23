#include <iostream>
#include <string>
using namespace std;

#include "acl/acl.h"
#include "shmem_api.h"

#include <gtest/gtest.h>
extern int test_gnpu_num;
extern int testFirstNpu;
extern void TestMutilTask(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int processCount);
extern void TestInit(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st);
extern void TestFinalize(aclrtStream stream, int device_id);

extern void PutOneNumDo(uint32_t block_dim, void* stream, uint8_t* gva, float val);

static int32_t TestScalarPutGet(aclrtStream stream, uint32_t rank_id, uint32_t rank_size)
{
    float *yHost;
    size_t inputSize = 1024 * sizeof(float);
    EXPECT_EQ(aclrtMallocHost((void **)(&yHost), inputSize), 0); // size = 1024

    uint32_t block_dim = 1;

    float value = 3.5f + (float)rank_id;
    void *ptr = shmem_malloc(1024);
    PutOneNumDo(block_dim, stream, (uint8_t *)ptr, value);
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    EXPECT_EQ(aclrtMemcpy(yHost, 1 * sizeof(float), ptr, 1 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST), 0);

    string pName = "[Process " + to_string(rank_id) + "] ";
    std::cout << pName << "-----[PUT]------ " << yHost[0] << " ----" << std::endl;

    // for gtest
    int32_t flag = 0;
    if (yHost[0] != (3.5f + (rank_id + rank_size - 1) % rank_size)) flag = 1;

    EXPECT_EQ(aclrtFreeHost(yHost), 0);
    return flag;
}

void TestShmemScalarP(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    int32_t device_id = rank_id % test_gnpu_num + testFirstNpu;
    aclrtStream stream;
    TestInit(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    int status = TestScalarPutGet(stream, rank_id, n_ranks);
    ASSERT_EQ(status, 0);

    std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    TestFinalize(stream, device_id);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

TEST(TestScalarPApi, TestShmemScalarP)
{
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemScalarP, local_mem_size, processCount);
}