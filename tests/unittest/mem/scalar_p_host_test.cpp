#include <iostream>
#include <string>
using namespace std;

#include "acl/acl.h"
#include "shmem_api.h"

#include <gtest/gtest.h>
extern int testGlobalRanks;
extern int testGNpuNum;
extern void TestMutilTask(std::function<void(int, int, uint64_t)> func, uint64_t localMemSize, int processCount);
extern void TestInit(int rankId, int nRanks, uint64_t localMemSize, aclrtStream *st);
extern void TestFinalize(aclrtStream stream, int deviceId);

extern void PutOneNumDo(uint32_t blockDim, void* stream, uint8_t* gva, float val);

static int32_t TestScalarPutGet(aclrtStream stream, uint32_t rankId, uint32_t rankSize)
{
    float *yHost;
    size_t inputSize = 1024 * sizeof(float);
    EXPECT_EQ(aclrtMallocHost((void **)(&yHost), inputSize), 0); // size = 1024

    uint32_t blockDim = 1;

    float value = 3.5f + (float)rankId;
    void *ptr = shmem_malloc(1024);
    PutOneNumDo(blockDim, stream, (uint8_t *)ptr, value);
    EXPECT_EQ(aclrtSynchronizeStream(stream), 0);
    sleep(2);

    EXPECT_EQ(aclrtMemcpy(yHost, 1 * sizeof(float), ptr, 1 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST), 0);

    string pName = "[Process " + to_string(rankId) + "] ";
    std::cout << pName << "-----[PUT]------ " << yHost[0] << " ----" << std::endl;

    // for gtest
    int32_t flag = 0;
    if (yHost[0] != (3.5f + (rankId + rankSize - 1) % rankSize)) flag = 1;

    EXPECT_EQ(aclrtFreeHost(yHost), 0);
    return flag;
}

void TestShmemScalarP(int rankId, int nRanks, uint64_t localMemSize)
{
    int32_t deviceId = rankId % testGNpuNum;
    aclrtStream stream;
    TestInit(rankId, nRanks, localMemSize, &stream);
    ASSERT_NE(stream, nullptr);

    int status = TestScalarPutGet(stream, rankId, nRanks);
    ASSERT_EQ(status, 0);

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    TestFinalize(stream, deviceId);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

TEST(TestScalarPApi, TestShmemScalarP)
{
    const int processCount = testGlobalRanks;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemScalarP, localMemSize, processCount);
}