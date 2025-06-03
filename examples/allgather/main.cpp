#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "shmem_api.h"

#define EXPECT_SUCCESS(status, exp)                                 \
    do {                                                            \
        if ((status) != 0) {                                        \
            std::cerr  << "Return err code: "  << status << ", at " \
            << __FILE__  << ":" << __LINE__ << std::endl;           \
            std::exit(EXIT_FAILURE);                                \
        }                                                           \
    } while (0)

int g_npus = 8;
const char* ipport;
int f_rank = 0;
int f_npu = 0;
extern void allgather_demo(uint32_t block_dim, void* stream, uint8_t* gva);

int test_shmem_team_all_gather(int rank_id, int n_ranks, uint64_t local_mem_size) {
    int32_t device_id = rank_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;
    
    EXPECT_SUCCESS(aclInit(nullptr), ACL_SUCCESS);
    EXPECT_SUCCESS(aclrtSetDevice(device_id), ACL_SUCCESS);
    EXPECT_SUCCESS(aclrtCreateStream(&stream), ACL_SUCCESS);

    shmem_init_attr_t* attributes;
    EXPECT_SUCCESS(shmem_set_attr(rank_id, n_ranks, local_mem_size, ipport, &attributes), SHMEM_SUCCESS);
    EXPECT_SUCCESS(shmem_init_attr(attributes), SHMEM_SUCCESS);

    void *ptr = shmem_malloc(1024);

    // Initialize data
    uint32_t trans_size = 16;
    std::vector<int32_t> input(trans_size, 0);
    for (int i = 0; i < trans_size; i++) {
        input[i] = (rank_id + 10);
    }

    EXPECT_SUCCESS(aclrtMemcpy(ptr + shmem_my_pe() * trans_size * sizeof(int32_t), trans_size * sizeof(int32_t), 
                          input.data(), trans_size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE), 0);

    // Execute AllGather
    allgather_demo(1, stream, (uint8_t *)ptr);
    EXPECT_SUCCESS(aclrtSynchronizeStream(stream), ACL_SUCCESS);

    // Check results
    int32_t *y_host;
    size_t input_size = n_ranks * trans_size * sizeof(int32_t);
    EXPECT_SUCCESS(aclrtMallocHost((void **) (&y_host), input_size), ACL_SUCCESS);
    EXPECT_SUCCESS(aclrtMemcpy(y_host, input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    
    for (int i = 0; i < n_ranks; i++) {
        EXPECT_SUCCESS(y_host[trans_size * i], 10 + i);
        std::cout << "rank: " << rank_id << " [";
        for (int j = 0; j < trans_size * n_ranks; j++) {
            std::cout << y_host[trans_size * i + j];
        }
        std::cout << "]" << std::endl;
    }

    EXPECT_SUCCESS(aclrtFreeHost(y_host), SHMEM_SUCCESS);
    shmem_free(ptr);
    EXPECT_SUCCESS(shmem_finalize(), SHMEM_SUCCESS);
    EXPECT_SUCCESS(aclrtDestroyStream(stream), ACL_SUCCESS);
    EXPECT_SUCCESS(aclrtResetDevice(device_id), ACL_SUCCESS);
    EXPECT_SUCCESS(aclFinalize(), ACL_SUCCESS);
    return 0

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
    EXPECT_SUCCESS(test_shmem_team_all_gather(rank_id, n_ranks, local_mem_size));

    if (status == 0) {
        std::cout << "[SUCCESS] demo run success in rank " << rank_id << std::endl;
    } else {
        std::cout << "[SUCCESS] demo run failed in rank " << rank_id << std::endl;
    }
    
    return 0;
}
