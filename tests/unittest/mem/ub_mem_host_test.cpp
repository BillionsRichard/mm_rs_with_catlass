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
extern void TestMutilTask(std::function<void(int, int, uint64_t)> func, uint64_t localMemSize, int processCount);
extern void TestInit(int rankId, int nRanks, uint64_t localMemSize, aclrtStream *st);
extern void TestFinalize(aclrtStream stream, int deviceId);

extern void TestUBPut(uint32_t blockDim, void* stream, uint8_t* gva, uint8_t* devPtr);
extern void TestUBGet(uint32_t blockDim, void* stream, uint8_t* gva, uint8_t* devPtr);

static void TestUBPutGet(aclrtStream stream, uint8_t *gva, uint32_t rankId, uint32_t rankSize)
{
    int totalSize = 512;
    size_t inputSize = totalSize * sizeof(float);
    
    std::vector<float> input(totalSize, 0);
    for (int i = 0; i < totalSize; i++) {
        input[i] = (rankId * 10);
    }
    
    void *devPtr;
    ASSERT_EQ(aclrtMalloc(&devPtr, inputSize, ACL_MEM_MALLOC_NORMAL_ONLY), 0);

    ASSERT_EQ(aclrtMemcpy(devPtr, inputSize, input.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);

    uint32_t blockDim = 1;
    void *ptr = shmem_malloc(totalSize * sizeof(float));
    TestUBPut(blockDim, stream, (uint8_t *)ptr, (uint8_t *)devPtr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    ASSERT_EQ(aclrtMemcpy(input.data(), inputSize, ptr, inputSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    string pName = "[Process " + to_string(rankId) + "] ";
    std::cout << pName;
    for (int i = 0; i < totalSize; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;

    TestUBGet(blockDim, stream, (uint8_t *)ptr, (uint8_t *)devPtr);
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
        int golden = rankId % rankSize;
        if (input[i] != golden * 10 + 55.0f) flag = 1;
    }
    ASSERT_EQ(flag, 0);
}

void TestShmemUBMem(int rankId, int nRanks, uint64_t localMemSize) {
    int32_t deviceId = rankId % testGNpuNum + testFirstNpu;
    aclrtStream stream;
    TestInit(rankId, nRanks, localMemSize, &stream);
    ASSERT_NE(stream, nullptr);

    TestUBPutGet(stream, (uint8_t *)shm::gState.heapBase, rankId, nRanks);
    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    TestFinalize(stream, deviceId);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

TEST(TestMemApi, TestShmemUBMem)
{   
    const int processCount = testGNpuNum;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemUBMem, localMemSize, processCount);
}