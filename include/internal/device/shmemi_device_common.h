#ifndef SHMEMI_DEVICE_COMMON_H
#define SHMEMI_DEVICE_COMMON_H

#include "macros.h"
#include "types.h"
#include "low_level_api/smem_shm_aicore_base_api.h"

SHMEM_AICORE_INLINE __gm__ ShmemDeviceHostState *getState() {
    return reinterpret_cast<__gm__ ShmemDeviceHostState *>(smem_shm_get_extra_context_addr());
}

SHMEM_AICORE_INLINE int getMyPe() {
    return getState()->mype;
}

SHMEM_AICORE_INLINE int getTotalPe() {
    return getState()->npes;
}

SHMEM_AICORE_INLINE uint64_t getMemSize() {
    return getState()->heapSize;
}

template<typename T>
SHMEM_AICORE_INLINE void store(__gm__ uint8_t *addr, T val) {
    *((__gm__ T *)addr) = val;
}

template<typename T>
SHMEM_AICORE_INLINE T load(__gm__ uint8_t *cache) {
    return *((__gm__ T *)cache);
}

SHMEM_AICORE_INLINE __gm__ uint8_t *ShmemiPtr(__gm__ uint8_t *local, int pe) {
    uint64_t shmSize = getMemSize();
    int myPe = getMyPe();

    uint64_t remote = reinterpret_cast<uint64_t>(local) + shmSize * (pe - myPe);
    return reinterpret_cast<__gm__ uint8_t*>(remote);
}

SHMEM_AICORE_INLINE void CubeGuard() {
#ifdef __DAV_C220_CUBE__
    mad(reinterpret_cast<__cc__ float *>((uint64_t)0),
        reinterpret_cast<__ca__ float *>((uint64_t)0),
        reinterpret_cast<__cb__ float *>((uint64_t)0),
        128, 0, 0, 0, 1, 0, 1);
#endif
}

SHMEM_AICORE_INLINE void VecGuard() {
#ifdef __DAV_C220_VEC__
    __ubuf__ float *buf = (__ubuf__ float *) get_imm(0);
    copy_ubuf_to_gm((__gm__ float *)0, buf, 0, 1, 0, 0, 0);
    copy_gm_to_ubuf(buf, (__gm__ float *)0, 0, 1, 0, 0, 0);
#endif
}

SHMEM_AICORE_INLINE void CVGuard() {
    CubeGuard();
    VecGuard();
}
#endif