#include <iostream>
#include <string>
#include <vector>
using namespace std;

#include "acl/acl.h"
#include "shmem_api.h"
#include "shmemi_host_common.h"

#include <gtest/gtest.h>
extern int testGNpuNum;
extern int testFirstNpu;
extern void TestMutilTask(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int processCount);
extern void TestInit(int rankId, int n_ranks, uint64_t local_mem_size, aclrtStream *st);
extern void TestFinalize(aclrtStream stream, int deviceId);

extern void TestPut(uint32_t blockDim, void* stream, uint8_t* gva, uint8_t* devPtr);
extern void TestGet(uint32_t blockDim, void* stream, uint8_t* gva, uint8_t* devPtr);

static void TestPutGet(aclrtStream stream, uint8_t *gva, uint32_t rankId, uint32_t rankSize)
{
    int totalSize = 16 * (int)rankSize;
    size_t inputSize = totalSize * sizeof(float);
    
    std::vector<float> input(totalSize, 0);
    for (int i = 0; i < 16; i++) {
        input[i] = (rankId + 10);
    }
    
    void *devPtr;
    ASSERT_EQ(aclrtMalloc(&devPtr, inputSize, ACL_MEM_MALLOC_NORMAL_ONLY), 0);

    ASSERT_EQ(aclrtMemcpy(devPtr, inputSize, input.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);

    uint32_t blockDim = 1;
    void *ptr = shmem_malloc(1024);
    TestPut(blockDim, stream, (uint8_t *)ptr, (uint8_t *)devPtr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    ASSERT_EQ(aclrtMemcpy(input.data(), inputSize, ptr, inputSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    string pName = "[Process " + to_string(rankId) + "] ";
    std::cout << pName;
    for (int i = 0; i < totalSize; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;

    TestGet(blockDim, stream, (uint8_t *)ptr, (uint8_t *)devPtr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    ASSERT_EQ(aclrtMemcpy(input.data(), inputSize, devPtr, inputSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    std::cout << pName;
    for (int i = 0; i < totalSize; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
    // for gtest
    int32_t flag = 0;
    for (int i = 0; i < totalSize; i++){
        int stage = i / 16;
        if (input[i] != (stage + 10)) flag = 1;
    }
    ASSERT_EQ(flag, 0);
}

void TestShmemMem(int rankId, int n_ranks, uint64_t local_mem_size) {
    int32_t deviceId = rankId % testGNpuNum + testFirstNpu;
    aclrtStream stream;
    TestInit(rankId, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    TestPutGet(stream, (uint8_t *)shm::gState.heap_base, rankId, n_ranks);
    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    TestFinalize(stream, deviceId);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

TEST(TestMemApi, TestShmemMem)
{   
    const int processCount = testGNpuNum;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemMem, local_mem_size, processCount);
}