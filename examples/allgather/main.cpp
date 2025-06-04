#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "shmem_api.h"

#define CHECK_SUCCESS(status, exp)                                 \
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
extern void allgather_demo(uint32_t block_dim, void* stream, uint8_t* gva, int elements);

int test_shmem_team_all_gather(int rank_id, int n_ranks, uint64_t local_mem_size) 
{
    // 初始化ACL和SHMEM
    int32_t device_id = rank_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;
    
    CHECK_SUCCESS(aclInit(nullptr), ACL_SUCCESS);
    CHECK_SUCCESS(aclrtSetDevice(device_id), ACL_SUCCESS);
    CHECK_SUCCESS(aclrtCreateStream(&stream), ACL_SUCCESS);

    shmem_init_attr_t* attributes;
    CHECK_SUCCESS(shmem_set_attr(rank_id, n_ranks, local_mem_size, ipport, &attributes), SHMEM_SUCCESS);
    CHECK_SUCCESS(shmem_init_attr(attributes), SHMEM_SUCCESS);

    void *ptr = shmem_malloc(1024);

    // 初始化数据
    uint32_t trans_size = 16;
    std::vector<int32_t> input(trans_size, 0);
    for (int i = 0; i < trans_size; i++) {
        input[i] = (rank_id + 10);
    }

    CHECK_SUCCESS(aclrtMemcpy(ptr + shmem_my_pe() * trans_size * sizeof(int32_t), trans_size * sizeof(int32_t), 
                          input.data(), trans_size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE), 0);

    // AllGather
    allgather_demo(1, stream, (uint8_t *)ptr, trans_size);
    CHECK_SUCCESS(aclrtSynchronizeStream(stream), ACL_SUCCESS);

    // 结果校验打印
    int32_t *y_host;
    size_t input_size = n_ranks * trans_size * sizeof(int32_t);
    CHECK_SUCCESS(aclrtMallocHost((void **) (&y_host), input_size), ACL_SUCCESS);
    CHECK_SUCCESS(aclrtMemcpy(y_host, input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    
    for (int i = 0; i < n_ranks; i++) {
        if (y_host[trans_size * i] != 10 + i) {
            std::cout << y_host[trans_size * i] << " != " << 10 + i << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    std::cout << "rank: " << rank_id << " [";
    for (int j = 0; j < trans_size * n_ranks; j++) {
        std::cout << y_host[j] << ", ";
    }
    std::cout << "]" << std::endl;
    // 去初始化
    CHECK_SUCCESS(aclrtFreeHost(y_host), SHMEM_SUCCESS);
    shmem_free(ptr);
    CHECK_SUCCESS(shmem_finalize(), SHMEM_SUCCESS);
    CHECK_SUCCESS(aclrtDestroyStream(stream), ACL_SUCCESS);
    CHECK_SUCCESS(aclrtResetDevice(device_id), ACL_SUCCESS);
    CHECK_SUCCESS(aclFinalize(), ACL_SUCCESS);
    return 0;
}

int main(int argc, char* argv[]) 
{
    int n_ranks = atoi(argv[1]);
    int rank_id = atoi(argv[2]);
    ipport = argv[3];
    g_npus = atoi(argv[4]);
    f_rank = atoi(argv[5]);
    f_npu = atoi(argv[6]);
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    CHECK_SUCCESS(test_shmem_team_all_gather(rank_id, n_ranks, local_mem_size), SHMEM_SUCCESS);
    std::cout << "[SUCCESS] demo run success in rank " << rank_id << std::endl;
    
    return 0;
}
