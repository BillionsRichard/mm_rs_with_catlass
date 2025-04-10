#include <iostream>
#include <string>

using namespace std;

#include "acl/acl.h"
#include "data_utils.h"

#include "smem.h"
#include "smem_shm.h"

extern void put_one_num_do(uint32_t blockDim, void* stream, uint8_t* gva, float val);

static uint32_t gNpuNum = 8;
static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

static int32_t TestScalarPutGet(aclrtStream stream, uint8_t *gva, uint32_t rankId, uint32_t rankSize)
{
    float *yHost;
    size_t inputSize = 1024 * sizeof(float);
    CHECK_ACL(aclrtMallocHost((void **)(&yHost), inputSize)); // size = 1024

    uint32_t blockDim = 1;

    float value = 3.5f + (float)rankId;
    put_one_num_do(blockDim, stream, gva, value);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    sleep(2);

    CHECK_ACL(aclrtMemcpy(yHost, 1 * sizeof(float), gva + rankId * gNpuMallocSpace, 1 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

    string pName = "[Process " + to_string(rankId) + "] ";
    std::cout << pName << "-----[PUT]------ " << yHost[0] << " ----" << std::endl;

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

    auto ret = smem_init(gNpuMallocSpace * rankSize, 0);
    if (ret != 0) {
        ERROR_LOG("[TEST] shm init failed, ret:%d, rank:%d \n", ret, rankId);
        return -1;
    }
    uint32_t flags = 0;
    void *gva = nullptr;
    smem_shm_t handle = smem_shm_create(0, ipport.c_str(), rankSize, rankId, deviceId, gNpuMallocSpace, SMEMS_DATA_OP_MTE, flags, &gva);
    if (handle == nullptr || gva == nullptr) {
        ERROR_LOG("[TEST] smem_shm_create failed, rank:%d \n", rankId);
        return -1;
    }
    WARN_LOG("[TEST] smem_shm_create gva %p, size %lu, rank:%d \n", gva, gNpuMallocSpace, rankId);

    TestScalarPutGet(stream, (uint8_t *)gva, rankId, rankSize);

    std::cout << "[TEST] begin to exit...... rank: " << rankId << std::endl;
    smem_shm_destroy(handle, flags);

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}