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

extern void TestUBNonContiguousPut(uint32_t blockDim, void* stream, uint8_t* gva, uint8_t* devPtr);
extern void TestUBNonContiguousGet(uint32_t blockDim, void* stream, uint8_t* gva, uint8_t* devPtr);

static void TestUBNonContiguousPutGet(aclrtStream stream, uint8_t *gva, uint32_t rankId, uint32_t rankSize)
{
    int row = 16;
    int col = 32;
    int totalSize = row * col;
    size_t inputSize = totalSize * sizeof(float);
    
    std::vector<float> input(totalSize, 0);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col / 2; j++) {
            input[i * col + j] = (rankId * 10);
        }
    }
    for (int i = 0; i < row; i++) {
        for (int j = col / 2; j < col; j++) {
            input[i * col + j] = (rankId * 10) + 1.0f;
        }
    }

    void *devPtr;
    ASSERT_EQ(aclrtMalloc(&devPtr, inputSize, ACL_MEM_MALLOC_NORMAL_ONLY), 0);
    ASSERT_EQ(aclrtMemcpy(devPtr, inputSize, input.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE), 0);

    uint32_t blockDim = 1;
    void *ptr = shmem_malloc(totalSize * sizeof(float));
    TestUBNonContiguousPut(blockDim, stream, (uint8_t *)ptr, (uint8_t *)devPtr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    ASSERT_EQ(aclrtMemcpy(input.data(), inputSize, ptr, inputSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    string pName = "[Process " + to_string(rankId) + "] ";
    if (rankId == 0) {
        std::cout << pName << std::endl;
        for (int i = 0; i < totalSize; i++) {
            std::cout << input[i] << " ";
            if (i % col == col - 1) {
                std::cout << std::endl;
            }
        }
    }

    TestUBNonContiguousGet(blockDim, stream, (uint8_t *)ptr, (uint8_t *)devPtr);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    ASSERT_EQ(aclrtMemcpy(input.data(), inputSize, devPtr, inputSize, ACL_MEMCPY_DEVICE_TO_HOST), 0);

    if (rankId == 0) {
        std::cout << pName << std::endl;
        for (int i = 0; i < totalSize; i++) {
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
            int golden = rankId % rankSize;
            if (input[i * col + j] != golden * 10) flag = 1;
        }
    }
    for (int i = 0; i < row; i++) {
        for (int j = col / 2; j < col; j++) {
            int golden = rankId % rankSize;
            if (input[i * col + j] != golden * 10 + 1.0f) flag = 1;
        }
    }
    ASSERT_EQ(flag, 0);
}

void TestShmemUBNonContiguous(int rankId, int nRanks, uint64_t localMemSize) {
    int32_t deviceId = rankId % testGNpuNum + testFirstNpu;
    aclrtStream stream;
    TestInit(rankId, nRanks, localMemSize, &stream);
    ASSERT_NE(stream, nullptr);

    TestUBNonContiguousPutGet(stream, (uint8_t *)shm::gState.heapBase, rankId, nRanks);
    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    TestFinalize(stream, deviceId);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

TEST(TestMemApi, TestShmemUBNonContiguous)
{   
    const int processCount = testGNpuNum;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemUBNonContiguous, localMemSize, processCount);
}