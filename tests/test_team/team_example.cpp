#include <iostream>
#include <cstdlib>
#include <string>
using namespace std;

#include <acl/acl.h>
#include "data_utils.h"

#include "smem.h"
#include "smem_shm.h"

#include "init.h"

static uint32_t gNpuNum = 8;
static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

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

    auto ret = smem_init(gNpuMallocSpace * rankSize, 0);
    if (ret != 0) {
        ERROR_LOG("[TEST] shm init failed, ret:%d, rankId:%d \n", ret, rankId);
        return -1;
    }
    uint32_t flags = 0;
    void *gva = nullptr;
    smem_shm_t handle = smem_shm_create(0, ipport.c_str(), rankSize, rankId, deviceId, gNpuMallocSpace, SMEMS_DATA_OP_MTE, flags, &gva);
    if (handle == nullptr || gva == nullptr) {
        ERROR_LOG("[TEST] smem_shm_create failed, rankId:%d \n", rankId);
        return -1;
    }
    WARN_LOG("[TEST] smem_shm_create gva %p, size %lu, rankId:%d \n", gva, gNpuMallocSpace, rankId);

    ShmemInit(rankId, rankSize);

    // #################### 子通信域切分测试 ############################
    ShmemTeam_t team_odd, team_even;
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

    // 保证前序子team创建完成，有个全局数组
    // xx_Barrier(SHMEM_TEAM_WORLD);
    sleep(2);

    start = 0;
    stride = 2;
    team_size = 4;
    ShmemTeamSplitStrided(SHMEM_TEAM_WORLD, start, stride, team_size, team_even);

    // 保证子team创建完成
    // xx_Barrier(SHMEM_TEAM_WORLD);
    sleep(2);

    std::cout << pFlag << "ShmemTeamTranslate_pe(team_even, 2, SHMEM_TEAM_WORLD): " << ShmemTeamTranslate_pe(team_even, 2, SHMEM_TEAM_WORLD) << std::endl;

    // #################### 相关资源释放 ##############################
    ShmemTeamDestroy(team_odd);
    ShmemTeamDestroy(team_even);

    ShmemFinalize();
    smem_shm_destroy(handle, flags);
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}