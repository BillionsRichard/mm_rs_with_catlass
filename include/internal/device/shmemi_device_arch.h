#ifndef SHMEMI_DEVICE_ARCH_H
#define SHMEMI_DEVICE_ARCH_H

#include "device/shmem_device_def.h"

SHMEM_DEVICE void DcciCacheline(__gm__ uint8_t * addr) {
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(addr);

    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

SHMEM_DEVICE void DcciEntireCache() {
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    
    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

#endif