#include "acl/acl.h"

#include "shmemi_device_intf.h"
#include "internal/device/shmemi_device_arch.h"

// kernels
template<typename T>
SHMEM_GLOBAL void k_memset(GM_ADDR array, int32_t len, T val, int32_t count) {
    auto tmp = (__gm__ T *) array;
    int32_t valid_count = count < len ? count : len;
    for (int32_t i = 0; i < valid_count; i++) {
        *tmp++ = val;
    }

    dcci_entire_cache();
} 

// interfaces
int32_t shmemi_memset(int32_t *array, int32_t len, int32_t val, int32_t count) {
    k_memset<int32_t><<<1, nullptr, nullptr>>>((uint8_t *)array, len, val, count);
    return aclrtSynchronizeStream(nullptr);
}