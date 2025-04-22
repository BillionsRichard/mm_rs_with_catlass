#include "acl/acl.h"
#include "macros.h"
#include "data_utils.h"
#include "shmem_sync.h"

// kernels
template<typename T>
SHMEM_AICORE_KERNEL void KMemset(GM_ADDR array, int len, T val) {
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