#include <iostream>
#include <cstdlib>
#include <string>
using namespace std;

#include <acl/acl.h>
#include "shmem_host_api.h"
#include "shmemi_host_intf.h"

#include <gtest/gtest.h>
extern int testGlobalRanks;
extern int testGNpuNum;
extern const char* testGlobalIpport;
extern void TestMutilTask(std::function<void(int, int, uint64_t)> func, uint64_t localMemSize, int processCount);

extern void GetDeviceState(uint32_t blockDim, void* stream, uint8_t* gva, ShmemTeam teamId);

static int32_t TestGetDeviceState(aclrtStream stream, uint8_t *gva, uint32_t rankId, uint32_t rankSize, ShmemTeam teamId)
{
    int *yHost;
    size_t inputSize = 1024 * sizeof(int);
    CHECK_ACL(aclrtMallocHost((void **)(&yHost), inputSize));       // size = 1024

    uint32_t blockDim = 1;
    void *ptr = ShmemMalloc(1024);
    GetDeviceState(blockDim, stream, (uint8_t *)ptr, teamId);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    sleep(2);

    CHECK_ACL(aclrtMemcpy(yHost, 5 * sizeof(int), ptr, 5 * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST));

    string pName = "[Process " + to_string(rankId) + "] ";
    std::cout << pName << "-----[PUT]------" << yHost[0] << " ---- " << yHost[1] << " ---- " << yHost[2] << " ---- " << yHost[3] << " ---- " << yHost[4] << std::endl;
    
    CHECK_ACL(aclrtFreeHost(yHost));
    return 0;
}

void TestShmemTeam(int rankId, int nRanks, uint64_t localMemSize) {
    int status = SHMEM_SUCCESS;
    std::cout << "[TEST] input rank_size: " << nRanks << " rank_id:" << rankId << " input_ip: " << testGlobalIpport << std::endl;

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
    ShmemSetAttr(rankId, nRanks, localMemSize, testGlobalIpport, &attributes);
    status = ShmemInit();
    EXPECT_EQ(status, SHMEM_SUCCESS);
    // #################### 子通信域切分测试 ############################
    ShmemTeam team_odd;
    int start = 1;
    int stride = 2;
    int team_size = 4;
    ShmemTeamSplitStrided(SHMEM_TEAM_WORLD, start, stride, team_size, team_odd);

    // #################### host侧取值测试 ##############################
    string pFlag = "[Process " + to_string(rankId) + "] ";
    std::cout << pFlag << "ShmemTeamNpes(team_odd): " << ShmemTeamNpes(team_odd) << std::endl;
    std::cout << pFlag << "ShmemTeamMype(team_odd): " << ShmemTeamMype(team_odd) << std::endl;
    std::cout << pFlag << "ShmemNpes(): " << ShmemNpes() << std::endl;
    std::cout << pFlag << "ShmemMype(): " << ShmemMype() << std::endl;
    sleep(2);

    // #################### device代码测试 ##############################

    status = TestGetDeviceState(stream, (uint8_t *)gState.heapBase, rankId, nRanks, team_odd);
    EXPECT_EQ(status, SHMEM_SUCCESS);

    // #################### 相关资源释放 ################################
    ShmemTeamDestroy(team_odd);

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    status = ShmemFinalize();
    EXPECT_EQ(status, SHMEM_SUCCESS);
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}



TEST(TestTeamApi, TestShmemTeam)
{   
    const int processCount = testGlobalRanks;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemTeam, localMemSize, processCount);
}