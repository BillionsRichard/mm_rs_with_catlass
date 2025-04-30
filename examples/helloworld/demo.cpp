#include <iostream>
#include <unistd.h>
#include <acl/acl.h>
#include "shmem_host_api.h"
int main(int argc, char* argv[]) 
{
    int nRanks = atoi(argv[1]);
    int rankId = atoi(argv[2]);
    const char* Ipport = argv[3];
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    int testGNpuNum = 8;
    std::cout << "[TEST] input rank_size: " << nRanks << " rank_id:" << rankId << " input_ip: " << Ipport << std::endl;
    uint32_t deviceId = rankId % testGNpuNum;
    int status = SHMEM_SUCCESS;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    ShmemInitAttrT* attributes;
    status = ShmemSetAttr(rankId, nRanks, localMemSize, Ipport, &attributes);
    if ( status != SHMEM_SUCCESS) {
        std::cout << "[ERROR] demo run failed!" << std::endl;
        std::exit(status);
    }
    status = ShmemInit();
    if ( status != SHMEM_SUCCESS) {
        std::cout << "[ERROR] demo run failed!" << std::endl;
        std::exit(status);
    }
    status = ShmemInitStatus();
    if (status == SHMEM_STATUS_IS_INITALIZED) {
        std::cout << "[SUCCESS] Shmem init success!" << std::endl;
    } else {
        std::cout << "[ERROR] demo run failed!" << std::endl;
        std::exit(status);
    }
    status = ShmemFinalize();
    if ( status != SHMEM_SUCCESS) {
        std::cout << "[ERROR] demo run failed!" << std::endl;
        std::exit(status);
    }
    CHECK_ACL(aclrtResetDevice(deviceId));
    aclFinalize();
    std::cout << "[SUCCESS] demo run success!" << std::endl;
}
