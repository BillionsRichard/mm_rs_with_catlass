#ifndef SHEMEI_P2P_H
#define SHEMEI_P2P_H

#include "internal/device/shmemi_device_common.h"

template<typename T>
SHMEM_DEVICE void ShmemiSignal(__gm__ uint8_t *addr, T val) {
    ShmemiStore<T>(addr, val);

    // flush data cache to GM after signal to ensure it is visiable to other ranks 
    DcciCacheline(addr);
}

template<typename T>
SHMEM_DEVICE void ShmemiSignal(__gm__ uint8_t *addr, int pe, T val) {
    __gm__ uint8_t *remote = ShmemiPtr(addr, pe);
    ShmemiSignal<T>(remote, val);
}

template<typename T>
SHMEM_DEVICE void ShmemiWait(__gm__ uint8_t *addr, T val) {
    while (ShmemiLoad<T>(addr) != val) {
      // always flush data cache to avoid reading staled data
      DcciCacheline(addr);
    }
}

#endif