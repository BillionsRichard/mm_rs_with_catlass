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
#endif