#include <iostream>
#include <cstdlib>
#include <string>
using namespace std;

#include <acl/acl.h>
#include "data_utils.h"

#include "shmem_api.h"

static uint32_t gNpuNum = 8;
static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

extern void GetDeviceState(uint32_t blockDim, void* stream, uint8_t* gva, ShmemTeam_t teamId);

static int32_t TestGetDeviceState(aclrtStream stream, uint8_t *gva, uint32_t rankId, uint32_t rankSize, ShmemTeam_t teamId)
{
    int *yHost;
    size_t inputSize = 1024 * sizeof(int);
    CHECK_ACL(aclrtMallocHost((void **)(&yHost), inputSize));       // size = 1024

    uint32_t blockDim = 1;

    GetDeviceState(blockDim, stream, gva, teamId);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    sleep(2);

    CHECK_ACL(aclrtMemcpy(yHost, 5 * sizeof(int), gva + rankId * gNpuMallocSpace, 5 * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST));

    string pName = "[Process " + to_string(rankId) + "] ";
    std::cout << pName << "-----[PUT]------" << yHost[0] << " ---- " << yHost[1] << " ---- " << yHost[2] << " ---- " << yHost[3] << " ---- " << yHost[4] << std::endl;
    
    CHECK_ACL(aclrtFreeHost(yHost));
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
    uint32_t flags = 0;
    ShmemInitAttr shmemInitAttr = CreateAttributes(0, ipport.c_str(), rankId, rankSize, deviceId, gNpuMallocSpace);

    ShmemInit(flags, &shmemInitAttr);
    // #################### 子通信域切分测试 ############################
    ShmemTeam_t team_odd;
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

    TestGetDeviceState(stream, (uint8_t *)shmemDeviceHostState.heapBase, rankId, rankSize, team_odd);

    // #################### 相关资源释放 ################################
    ShmemTeamDestroy(team_odd);

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    ShmemFinalize(flags);
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}