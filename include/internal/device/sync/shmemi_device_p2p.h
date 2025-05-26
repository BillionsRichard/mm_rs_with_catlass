#ifndef SHEMEI_P2P_H
#define SHEMEI_P2P_H

#include "internal/device/shmemi_device_common.h"

template<typename T>
SHMEM_DEVICE void shmemi_signal(__gm__ uint8_t *addr, T val) {
    shmemi_store<T>(addr, val);

    // flush data cache to GM after signal to ensure it is visiable to other ranks 
    dcci_cacheline(addr);
}

template<typename T>
SHMEM_DEVICE void shmemi_signal(__gm__ uint8_t *addr, int pe, T val) {
    __gm__ uint8_t *remote = shmemi_ptr(addr, pe);
    shmemi_signal<T>(remote, val);
}

template<typename T>
SHMEM_DEVICE void shmemi_wait(__gm__ uint8_t *addr, T val) {
    while (shmemi_load<T>(addr) != val) {
      // always flush data cache to avoid reading staled data
      dcci_cacheline(addr);
    }
}

#endif