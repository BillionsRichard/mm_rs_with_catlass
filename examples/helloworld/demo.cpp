#include <iostream>
#include <unistd.h>
#include <cstring>
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
    int n_ranks = atoi(argv[1]);
    int rank_id = atoi(argv[2]);
    size_t ip_len = strlen(argv[3]);
    char* Ipport = new char[ip_len + 1];
    errno_t ret = strcpy_s(Ipport, ip_len + 1, argv[3]);
    if (ret != EOK) {
        std::cout << "[ERROR] demo run failed!" << std::endl;
        std::exit(1);
    }
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    int test_gnpu_num = 8;
    std::cout << "[TEST] input rank_size: " << n_ranks << " rank_id:" << rank_id << " input_ip: " << Ipport << std::endl;
    uint32_t device_id = rank_id % test_gnpu_num;
    int status = SHMEM_SUCCESS;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));
    shmem_init_attr_t *attributes;
    status = shmem_set_attr(rank_id, n_ranks, local_mem_size, Ipport, &attributes);
    delete[] Ipport;
    if ( status != SHMEM_SUCCESS) {
        std::cout << "[ERROR] demo run failed!" << std::endl;
        std::exit(status);
    }
    status = shmem_init_attr(attributes);
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
    CHECK_ACL(aclrtResetDevice(device_id));
    aclFinalize();
    std::cout << "[SUCCESS] demo run success!" << std::endl;
}
