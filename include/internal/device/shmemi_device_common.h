#ifndef SHMEMI_DEVICE_COMMON_H
#define SHMEMI_DEVICE_COMMON_H

#include "shmemi_device_arch.h"
#include "shmemi_device_def.h"

#include "lowlevel/smem_shm_aicore_base_api.h"

SHMEM_DEVICE __gm__ ShmemiDeviceHostState *ShmemiGetState() {
    return reinterpret_cast<__gm__ ShmemiDeviceHostState *>(smem_shm_get_extra_context_addr());
}

SHMEM_DEVICE int ShmemiGetMyPe() {
    return ShmemiGetState()->mype;
}

SHMEM_DEVICE int ShmemiGetTotalPe() {
    return ShmemiGetState()->npes;
}

SHMEM_DEVICE uint64_t ShmemiGetHeapSize() {
    return ShmemiGetState()->heapSize;
}

template<typename T>
SHMEM_DEVICE void ShmemiStore(__gm__ uint8_t *addr, T val) {
    *((__gm__ T *)addr) = val;
}

template<typename T>
SHMEM_DEVICE T ShmemiLoad(__gm__ uint8_t *cache) {
    return *((__gm__ T *)cache);
}

SHMEM_DEVICE __gm__ uint8_t *ShmemiPtr(__gm__ uint8_t *local, int pe) {
    uint64_t shmSize = ShmemiGetHeapSize();
    int myPe = ShmemiGetMyPe();

    uint64_t remote = reinterpret_cast<uint64_t>(local) + shmSize * (pe - myPe);
    return reinterpret_cast<__gm__ uint8_t*>(remote);
}

SHMEM_DEVICE void CubeGuard() {
#ifdef __DAV_C220_CUBE__
    mad(reinterpret_cast<__cc__ float *>((uint64_t)0),
        reinterpret_cast<__ca__ float *>((uint64_t)0),
        reinterpret_cast<__cb__ float *>((uint64_t)0),
        128, 0, 0, 0, 1, 0, 1);
#endif
}

SHMEM_DEVICE void VecGuard() {
#ifdef __DAV_C220_VEC__
    __ubuf__ float *buf = (__ubuf__ float *) get_imm(0);
    copy_ubuf_to_gm((__gm__ float *)0, buf, 0, 1, 0, 0, 0);
    copy_gm_to_ubuf(buf, (__gm__ float *)0, 0, 1, 0, 0, 0);
#endif
}

SHMEM_DEVICE void CVGuard() {
    CubeGuard();
    VecGuard();
}
#endif