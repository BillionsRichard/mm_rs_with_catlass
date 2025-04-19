#ifndef SHEMEI_P2P_H
#define SHEMEI_P2P_H

#include "internal/device/shmemi_device_common.h"

template<typename T>
SHMEM_AICORE_INLINE void ShmemiSignal(__gm__ uint8_t *addr, int pe, T val) {
    __gm__ uint8_t *remote = ShmemiPtr(addr, pe);
    store<T>(remote, val);

    // flush data cache to GM after signal to ensure it is visiable to other ranks 
    DcciCacheline(remote);
}

template<typename T>
SHMEM_AICORE_INLINE void ShmemiWait(__gm__ uint8_t *addr, T val) {
    while (load<T>(addr) != val) {
      // always flush data cache to avoid reading staled data
      DcciCacheline(addr);
    }
}

#endif