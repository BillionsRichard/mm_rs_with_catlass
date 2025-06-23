#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "shmem_api.h"
#include "shmemi_host_common.h"

#include <gtest/gtest.h>
using namespace std;
extern int test_gnpu_num;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int processCount);
extern void test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st);
extern void test_finalize(aclrtStream stream, int device_id);

extern void TestUBNonContiguousPut(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev_ptr);
extern void test_ub_non_contiguous_get(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev_ptr);

static void TestUBNonContiguousPutGet(aclrtStream stream, uint8_t *gva, uint32_t rank_id, uint32_t rank_size)
{
    int row = 16;
    int col = 32;
    int total_size = row * col;
    size_t input_size = total_size * sizeof(float);
    
    std::vector<float> input(total_size, 0);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col / 2; j++) {
            input[i * col + j] = (rank_id * 10);
        }
    }
    for (int i = 0; i < row; i++) {
        for (int j = col / 2; j < col; j++) {
            input[i * col + j] = (rank_id * 10) + 1.0f;
        }
    }

    void *dev_ptr;
    ASSERT_EQ(aclrtMalloc(&dev_ptr, input_size, ACL_MEM_MALLOC_NORMAL_ONLY), 0);
    ASSERT_EQ(aclrtMemcpy(dev_ptr, input_size, input.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 0);

    uint32_t block_dim = 1;
    void *ptr = shmem_malloc(total_size * sizeof(float));
    TestUBNonContiguousPut(block_dim, stream, (uint8_t *)ptr, (uint8_t *)dev_ptr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    ASSERT_EQ(aclrtMemcpy(input.data(), input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    string p_name = "[Process " + to_string(rank_id) + "] ";
    if (rank_id == 0) {
        std::cout << p_name << std::endl;
        for (int i = 0; i < total_size; i++) {
            std::cout << input[i] << " ";
            if (i % col == col - 1) {
                std::cout << std::endl;
            }
        }
    }

    test_ub_non_contiguous_get(block_dim, stream, (uint8_t *)ptr, (uint8_t *)dev_ptr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    ASSERT_EQ(aclrtMemcpy(input.data(), input_size, dev_ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    if (rank_id == 0) {
        std::cout << p_name << std::endl;
        for (int i = 0; i < total_size; i++) {
            std::cout << input[i] << " ";
            if (i % col == col - 1) {
                std::cout << std::endl;
            }
        }
    }

    // for gtest
    int32_t flag = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col / 2; j++) {
            int golden = rank_id % rank_size;
            if (input[i * col + j] != golden * 10) flag = 1;
        }
    }
    for (int i = 0; i < row; i++) {
        for (int j = col / 2; j < col; j++) {
            int golden = rank_id % rank_size;
            if (input[i * col + j] != golden * 10 + 1.0f) flag = 1;
        }
    }
    ASSERT_EQ(flag, 0);
}

void test_shmem_ub_non_contiguous(int rank_id, int n_ranks, uint64_t local_mem_size) {
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    TestUBNonContiguousPutGet(stream, (uint8_t *)shm::g_state.heap_base, rank_id, n_ranks);
    std::cout << "[TEST] begin to exit...... rank_id: " << rank_id << std::endl;
    test_finalize(stream, device_id);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

TEST(TestMemApi, TestShmemUBNonContiguous)
{   
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_shmem_ub_non_contiguous, local_mem_size, processCount);
}