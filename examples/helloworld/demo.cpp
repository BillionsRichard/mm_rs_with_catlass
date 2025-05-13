#include <iostream>
#include <unistd.h>
#include <acl/acl.h>
#include "shmem_api.h"

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << "[ERROR]" << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

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
    shmem_init_attr_t *attributes;
    status = shmem_set_attr(rankId, nRanks, localMemSize, Ipport, &attributes);
    if ( status != SHMEM_SUCCESS) {
        std::cout << "[ERROR] demo run failed!" << std::endl;
        std::exit(status);
    }
    status = shmem_init(attributes);
    if ( status != SHMEM_SUCCESS) {
        std::cout << "[ERROR] demo run failed!" << std::endl;
        std::exit(status);
    }
    status = shmem_init_status();
    if (status == SHMEM_STATUS_IS_INITALIZED) {
        std::cout << "[SUCCESS] Shmem init success!" << std::endl;
    } else {
        std::cout << "[ERROR] demo run failed!" << std::endl;
        std::exit(status);
    }
    status = shmem_finalize();
    if ( status != SHMEM_SUCCESS) {
        std::cout << "[ERROR] demo run failed!" << std::endl;
        std::exit(status);
    }
    CHECK_ACL(aclrtResetDevice(deviceId));
    aclFinalize();
    std::cout << "[SUCCESS] demo run success!" << std::endl;
}
