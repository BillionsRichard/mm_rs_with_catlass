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

extern void PutOneNumDo(uint32_t blockDim, void* stream, uint8_t* gva, float val);

static int32_t TestScalarPutGet(aclrtStream stream, uint8_t *gva, uint32_t rankId, uint32_t rankSize)
{
    float *yHost;
    size_t inputSize = 1024 * sizeof(float);
    CHECK_ACL(aclrtMallocHost((void **)(&yHost), inputSize)); // size = 1024

    uint32_t blockDim = 1;

    float value = 3.5f + (float)rankId;
    void *ptr = ShmemMalloc(1024);
    PutOneNumDo(blockDim, stream, (uint8_t *)ptr, value);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    sleep(2);

    CHECK_ACL(aclrtMemcpy(yHost, 1 * sizeof(float), ptr, 1 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

    string pName = "[Process " + to_string(rankId) + "] ";
    std::cout << pName << "-----[PUT]------ " << yHost[0] << " ----" << std::endl;

    // for gtest
    int32_t flag = 0;
    if (yHost[0] != (3.5f + (rankId + rankSize - 1) % rankSize)) flag = 1;

    CHECK_ACL(aclrtFreeHost(yHost));
    return flag;
}

void TestShmemScalarP(int rankId, int nRanks, uint64_t localMemSize)
{
    int status = SHMEM_SUCCESS;
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
    ShmemInitAttrT* attributes;
    ShmemSetAttr(rankId, nRanks, localMemSize, test_global_ipport, &attributes);
    status = ShmemInit();
    EXPECT_EQ(status, SHMEM_SUCCESS);

    status = TestScalarPutGet(stream, (uint8_t *)shmemDeviceHostState.heapBase, rankId, nRanks);
    EXPECT_EQ(status, SHMEM_SUCCESS);

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    status = ShmemFinalize();
    EXPECT_EQ(status, SHMEM_SUCCESS);
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
}

TEST(TestScalarPApi, TestShmemScalarP)
{   
    const int processCount = testGlobalRanks;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemScalarP, localMemSize, processCount);
}