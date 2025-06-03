#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmem_api.h"

int g_npus = 8;
const char* ipport;
int f_rank = 0;
int f_npu = 0;
extern void allgather_demo(uint32_t block_dim, void* stream, uint8_t* gva);

void test_shmem_team_all_gather(int rank_id, int n_ranks, uint64_t local_mem_size) {
    int32_t device_id = rank_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    EXPECT_EQ(status = aclrtCreateStream(&stream), 0);

    shmem_init_attr_t* attributes;
    shmem_set_attr(rank_id, n_ranks, local_mem_size, ipport, &attributes);
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, 0);
    ASSERT_NE(stream, nullptr);

    void *ptr = shmem_malloc(1024);

    // Initialize data
    uint32_t trans_size = 16;
    std::vector<int32_t> input(trans_size, 0);
    for (int i = 0; i < trans_size; i++) {
        input[i] = (rank_id + 10);
    }

    ASSERT_EQ(aclrtMemcpy(ptr + shmem_my_pe() * trans_size * sizeof(int32_t), trans_size * sizeof(int32_t), 
                          input.data(), trans_size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE), 0);

    // Execute AllGather
    allgather_demo(1, stream, (uint8_t *)ptr);
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);

    // Check results
    int32_t *y_host;
    size_t input_size = n_ranks * trans_size * sizeof(int32_t);
    EXPECT_EQ(aclrtMallocHost((void **) (&y_host), input_size), 0);
    EXPECT_EQ(aclrtMemcpy(y_host, input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    
    for (int i = 0; i < n_ranks; i++) {
        EXPECT_EQ(y_host[trans_size * i], 10 + i);
    }
    
    EXPECT_EQ(aclrtFreeHost(y_host), 0);
    
    shmem_free(ptr);
    status = shmem_finalize();
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclrtDestroyStream(stream), 0);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);

}

int main(int argc, char* argv[]) 
{
    int n_ranks = atoi(argv[1]);
    int rank_id = atoi(argv[2]);
    ipport = argv[3];
    g_npus = atoi(argv[4]);
    f_rank = atoi(argv[5]);
    f_npu = atoi(argv[6]);
    uint64_t local_mem_size = 1024UL * 1024UL *1024;
    test_shmem_team_all_gather(rank_id, n_ranks, local_mem_size);


    std::cout << "[SUCCESS] demo run success!" << std::endl;
    return 0;
}
