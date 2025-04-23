#include <iostream>
#include <string>
using namespace std;

#include "acl/acl.h"
#include "data_utils.h"

#include "shmem_api.h"

#include <gtest/gtest.h>
extern int testGlobalRanks;
extern int testGNpuNum;
extern const char* testGlobalIpport;
extern void TestMutilTask(std::function<void(int, int, uint64_t)> func, uint64_t localMemSize, int processCount);

extern void TestPut(uint32_t blockDim, void* stream, uint8_t* gva, uint8_t* devPtr);
extern void TestGet(uint32_t blockDim, void* stream, uint8_t* gva, uint8_t* devPtr);

static int32_t TestPutGet(aclrtStream stream, uint8_t *gva, uint32_t rankId, uint32_t rankSize)
{
    int totalSize = 16 * (int)rankSize;
    size_t inputSize = totalSize * sizeof(float);
    
    std::vector<float> input(totalSize, 0);
    for (int i = 0; i < 16; i++) {
        input[i] = (rankId + 10);
    }
    
    void *devPtr;
    CHECK_ACL(aclrtMalloc(&devPtr, inputSize, ACL_MEM_MALLOC_NORMAL_ONLY));

    CHECK_ACL(aclrtMemcpy(devPtr, inputSize, input.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint32_t blockDim = 1;
    void *ptr = ShmemMalloc(1024);
    TestPut(blockDim, stream, (uint8_t *)ptr, (uint8_t *)devPtr);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    sleep(2);

    CHECK_ACL(aclrtMemcpy(input.data(), inputSize, ptr, inputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    string pName = "[Process " + to_string(rankId) + "] ";
    std::cout << pName;
    for (int i = 0; i < totalSize; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;

    TestGet(blockDim, stream, (uint8_t *)ptr, (uint8_t *)devPtr);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    sleep(2);

    CHECK_ACL(aclrtMemcpy(input.data(), inputSize, devPtr, inputSize, ACL_MEMCPY_DEVICE_TO_HOST));

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
    return flag;
}

void TestShmemMem(int rankId, int nRanks, uint64_t localMemSize) {
    int status = SHMEM_SUCCESS;
    std::cout << "[TEST] input rank_size: " << nRanks << " rank_id:" << rankId << " input_ip: " << test_global_ipport << std::endl;

    if (nRanks != (nRanks & (~(nRanks - 1)))) {
        std::cout << "[TEST] input rank_size: "<< nRanks << " is not the power of 2" << std::endl;
        status = ERROR_INVALID_VALUE;
    }
    EXPECT_EQ(status, SHMEM_SUCCESS);
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = rankId % testGNpuNum;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;

    CHECK_ACL(aclrtCreateStream(&stream));
    status = ShmemInit(rankId, nRanks, localMemSize);
    EXPECT_EQ(status, SHMEM_SUCCESS);
    status = TestPutGet(stream, (uint8_t *)shmemDeviceHostState.heapBase, rankId, nRanks);
    EXPECT_EQ(status, SHMEM_SUCCESS);

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    status = ShmemFinalize();
    EXPECT_EQ(status, SHMEM_SUCCESS);
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
}

TEST(TestMemApi, TestShmemMem)
{   
    const int processCount = testGlobalRanks;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemMem, localMemSize, processCount);
}