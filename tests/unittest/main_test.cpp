#include <gtest/gtest.h>
#include <iostream>
#include "acl/acl.h"
#include "shmem_api.h"

int testGlobalRanks;
int test_gnpu_num;
const char* test_global_ipport;
int testFirstRank;
int testFirstNpu;

void TestInit(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st)
{
    *st = nullptr;
    int status = 0;
    if (n_ranks != (n_ranks & (~(n_ranks - 1)))) {
        std::cout << "[TEST] input rank_size: "<< n_ranks << " is not the power of 2" << std::endl;
        status = -1;
    }
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclInit(nullptr), 0);
    int32_t device_id = rank_id % test_gnpu_num + testFirstNpu;
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    aclrtStream stream = nullptr;
    EXPECT_EQ(status = aclrtCreateStream(&stream), 0);

    shmem_init_attr_t* attributes;
    shmem_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, 0);
    *st = stream;
}

void TestFinalize(aclrtStream stream, int device_id)
{
    int status = shmem_finalize();
    EXPECT_EQ(status, 0);
    EXPECT_EQ(aclrtDestroyStream(stream), 0);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
}

void TestMutilTask(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int processCount){
    pid_t pids[processCount];
    int status[processCount];
    for (int i = 0; i < processCount; ++i) {
        pids[i] = fork();
        if (pids[i] < 0) {
            std::cout << "fork failed ! " << pids[i] << std::endl;
        } else if (pids[i] == 0) {
            func(i + testFirstRank, testGlobalRanks, local_mem_size);
            exit(0);
        }
    }
    for (int i = 0; i < processCount; ++i) {
        waitpid(pids[i], &status[i], 0);
        if (WIFEXITED(status[i]) && WEXITSTATUS(status[i]) != 0) {
            FAIL();
        }
    }
}

int main(int argc, char** argv) {
    testGlobalRanks = std::atoi(argv[1]);
    test_global_ipport = argv[2];
    test_gnpu_num = std::atoi(argv[3]);
    testFirstRank = std::atoi(argv[4]);
    testFirstNpu = std::atoi(argv[5]);

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}