#include <iostream>
#include <string>
using namespace std;

#include "acl/acl.h"
#include "data_utils.h"

#include "shmem_api.h"

extern void TestPut(uint32_t blockDim, void* stream, uint8_t* gva, uint8_t* devPtr);
extern void TestGet(uint32_t blockDim, void* stream, uint8_t* gva, uint8_t* devPtr);

static uint32_t gNpuNum = 8;
static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

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
    TestPut(blockDim, stream, gva + rankId * gNpuMallocSpace, (uint8_t *)devPtr);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    sleep(2);

    CHECK_ACL(aclrtMemcpy(input.data(), inputSize, gva + rankId * gNpuMallocSpace, inputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    string pName = "[Process " + to_string(rankId) + "] ";
    std::cout << pName;
    for (int i = 0; i < totalSize; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;

    TestGet(blockDim, stream, gva + rankId * gNpuMallocSpace, (uint8_t *)devPtr);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    sleep(2);

    CHECK_ACL(aclrtMemcpy(input.data(), inputSize, devPtr, inputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    std::cout << pName;
    for (int i = 0; i < totalSize; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}

int main(int argc, char* argv[]) 
{
    int rankSize = atoi(argv[1]);
    int rankId = atoi(argv[2]);
    std::string ipport = argv[3];
    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << " input_ip: " << ipport << std::endl;

    if (rankSize != (rankSize & (~(rankSize - 1)))) {
        std::cout << "[TEST] input rank_size: "<< rankSize << " is not the power of 2" << std::endl;
        return -1;
    }

    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = rankId % gNpuNum;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;

    CHECK_ACL(aclrtCreateStream(&stream));
    ShmemInit(rankId, rankSize, gNpuMallocSpace);

    TestPutGet(stream, (uint8_t *)shmemDeviceHostState.heapBase, rankId, rankSize);

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    ShmemFinalize();
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}