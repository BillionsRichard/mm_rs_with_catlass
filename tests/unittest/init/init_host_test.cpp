#include <iostream>
#include <unistd.h>
#include <acl/acl.h>
#include "shmem_api.h"
#include "shmemi_host_common.h"
#include <gtest/gtest.h>
extern int test_gnpu_num;
extern const char* test_global_ipport;
extern int testFirstNpu;
extern void TestMutilTask(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int processCount);

namespace shm {
extern shmem_init_attr_t gAttr;
}

void TestShmemInit(int rank_id, int n_ranks, uint64_t local_mem_size) {
    uint32_t device_id = rank_id % test_gnpu_num + testFirstNpu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_init_attr_t* attributes;
    shmem_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(shm::gState.mype, rank_id);
    EXPECT_EQ(shm::gState.npes, n_ranks);
    EXPECT_NE(shm::gState.heap_base, nullptr);
    EXPECT_NE(shm::gState.p2p_heap_base[rank_id], nullptr);
    EXPECT_EQ(shm::gState.heap_size, local_mem_size + SHMEM_EXTRA_SIZE);
    EXPECT_NE(shm::gState.team_pools[0], nullptr);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_IS_INITALIZED);
    status = shmem_finalize();
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void TestShmemInitAttrT(int rank_id, int n_ranks, uint64_t local_mem_size) {
    uint32_t device_id = rank_id % test_gnpu_num + testFirstNpu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);

    shmem_init_attr_t* attributes = new shmem_init_attr_t{rank_id, n_ranks, test_global_ipport, local_mem_size, {0, SHMEM_DATA_OP_MTE, 120, 120, 120}};
    status = shmem_init_attr(attributes);

    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(shm::gState.mype, rank_id);
    EXPECT_EQ(shm::gState.npes, n_ranks);
    EXPECT_NE(shm::gState.heap_base, nullptr);
    EXPECT_NE(shm::gState.p2p_heap_base[rank_id], nullptr);
    EXPECT_EQ(shm::gState.heap_size, local_mem_size + SHMEM_EXTRA_SIZE);
    EXPECT_NE(shm::gState.team_pools[0], nullptr);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_IS_INITALIZED);
    status = shmem_finalize();
    delete attributes;
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void TestShmemInitInvalidRankId(int rank_id, int n_ranks, uint64_t local_mem_size) {
    int erankId = -1;
    uint32_t device_id = rank_id % test_gnpu_num + testFirstNpu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_init_attr_t* attributes;
    shmem_set_attr(erankId, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, SHMEM_INVALID_VALUE);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void TestShmemInitRankIdOverSize(int rank_id, int n_ranks, uint64_t local_mem_size) {
    uint32_t device_id = rank_id % test_gnpu_num + testFirstNpu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_init_attr_t* attributes;
    shmem_set_attr(rank_id + n_ranks, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, SHMEM_INVALID_PARAM);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void TestShmemInitZeroMem(int rank_id, int n_ranks, uint64_t local_mem_size) {
    //local_mem_size = 0
    uint32_t device_id = rank_id % test_gnpu_num + testFirstNpu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_init_attr_t* attributes;
    shmem_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, SHMEM_INVALID_VALUE);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void TestShmemInitInvalidMem(int rank_id, int n_ranks, uint64_t local_mem_size) {
    //local_mem_size = invalid
    uint32_t device_id = rank_id % test_gnpu_num + testFirstNpu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_init_attr_t* attributes;
    shmem_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, SHMEM_SMEM_ERROR);
    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_NOT_INITALIZED);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

void TestShmemSetConfig(int rank_id, int n_ranks, uint64_t local_mem_size) {
    uint32_t device_id = rank_id % test_gnpu_num + testFirstNpu;
    int status = SHMEM_SUCCESS;
    EXPECT_EQ(aclInit(nullptr), 0);
    EXPECT_EQ(status = aclrtSetDevice(device_id), 0);
    shmem_init_attr_t* attributes;
    shmem_set_attr(rank_id, n_ranks, local_mem_size, test_global_ipport, &attributes);

    shmem_set_data_op_engine_type(attributes, SHMEM_DATA_OP_MTE);
    shmem_set_timeout(attributes, 50);
    EXPECT_EQ(shm::gAttr.option_attr.control_operation_timeout, 50);
    EXPECT_EQ(shm::gAttr.option_attr.data_op_engine_type, SHMEM_DATA_OP_MTE);
    
    status = shmem_init_attr(attributes);
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(shm::gState.mype, rank_id);
    EXPECT_EQ(shm::gState.npes, n_ranks);
    EXPECT_NE(shm::gState.heap_base, nullptr);
    EXPECT_NE(shm::gState.p2p_heap_base[rank_id], nullptr);
    EXPECT_EQ(shm::gState.heap_size, local_mem_size + SHMEM_EXTRA_SIZE);
    EXPECT_NE(shm::gState.team_pools[0], nullptr);

    EXPECT_EQ(shm::gAttr.option_attr.control_operation_timeout, 50);
    EXPECT_EQ(shm::gAttr.option_attr.data_op_engine_type, SHMEM_DATA_OP_MTE);

    status = shmem_init_status();
    EXPECT_EQ(status, SHMEM_STATUS_IS_INITALIZED);
    status = shmem_finalize();
    EXPECT_EQ(status, SHMEM_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(device_id), 0);
    EXPECT_EQ(aclFinalize(), 0);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

TEST(TestInitAPI, TestShmemInit)
{   
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemInit, local_mem_size, processCount);
}

TEST(TestInitAPI, TestShmemInitAttrT)
{   
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemInitAttrT, local_mem_size, processCount);
}

TEST(TestInitAPI, TestShmemInitErrorInvalidRankId)
{   
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemInitInvalidRankId, local_mem_size, processCount);
}

TEST(TestInitAPI, TestShmemInitErrorRankIdOversize)
{   
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemInitRankIdOverSize, local_mem_size, processCount);
}

TEST(TestInitAPI, TestShmemInitErrorZeroMem)
{   
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 0;
    TestMutilTask(TestShmemInitZeroMem, local_mem_size, processCount);
}

TEST(TestInitAPI, TestShmemInitErrorInvalidMem)
{   
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL;
    TestMutilTask(TestShmemInitInvalidMem, local_mem_size, processCount);
}

TEST(TestInitAPI, TestSetConfig)
{   
    const int processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemSetConfig, local_mem_size, processCount);
}