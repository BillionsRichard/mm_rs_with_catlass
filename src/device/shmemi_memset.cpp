#include "acl/acl.h"

#include "shmemi_device_intf.h"
#include "internal/device/shmemi_device_arch.h"

// kernels
template<typename T>
SHMEM_GLOBAL void KMemset(GM_ADDR array, int32_t len, T val) {
    auto tmp = (__gm__ T *) array;
    for (int32_t i = 0; i < len; i++) {
        *tmp++ = val;
    }

    DcciEntireCache();
} 

// interfaces
int32_t ShmemiMemset(int32_t *array, int32_t len, int32_t val) {
    KMemset<int32_t><<<1, nullptr, nullptr>>>((uint8_t *)array, len, val);
    return aclrtSynchronizeStream(nullptr);
}