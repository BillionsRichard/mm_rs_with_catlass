#include "acl/acl.h"

#include "shmemi_host_intf.h"
#include "shmem_device_api.h"

// kernels
template<typename T>
SHMEM_GLOBAL void KMemset(GM_ADDR array, int len, T val) {
    auto tmp = (__gm__ T*) array;
    for (int i = 0; i < len; i++) {
        *tmp++ = val;
    }

    DcciEntireCache();
} 

// interfaces
void ShmemiMemset(int* array, int len, int val) {
    KMemset<int><<<1, nullptr, nullptr>>>((uint8_t *)array, len, val);
    CHECK_ACL(aclrtSynchronizeStream(nullptr));
}