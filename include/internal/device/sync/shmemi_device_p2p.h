/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHEMEI_P2P_H
#define SHEMEI_P2P_H

#include "shmemi_device_quiet.h"

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
    do {
        // always flush data cache to avoid reading staled data
        dcci_cacheline(addr);
    } while (shmemi_load<T>(addr) != val);
}

SHMEM_DEVICE void shmemi_signal_set(__gm__ int32_t *addr, int pe, int32_t val) {
    shmemi_signal<int32_t>((__gm__ uint8_t *)addr, pe, val);
}

SHMEM_DEVICE void shmemi_signal_add(__gm__ int32_t *addr, int pe, int32_t val) {
    // ensure previous atomic operations end
    dcci_atomic();
    dsb_all();

    // atomic add
    set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);
    st_atomic<int32_t>(val, shmemi_ptr(addr, pe));
    dcci_atomic();
}

// Atomicity of SHMEM_SIGNAL_SET not guaranteed
SHMEM_DEVICE void shmemix_signal_op(__gm__ int32_t *sig_addr, int32_t signal, int sig_op, int pe) {
    shmemi_quiet();

    switch (sig_op) {
        case SHMEM_SIGNAL_SET:
            shmemi_signal_set(sig_addr, pe, signal);
            break;
        case SHMEM_SIGNAL_ADD:
            shmemi_signal_add(sig_addr, pe, signal);
            break;
    }
}

SHMEM_DEVICE int32_t shmemi_signal_wait_until_eq(__gm__ int32_t *sig_addr, int32_t cmp_val) {
    int32_t ret;
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while ((ret = *sig_addr) != cmp_val);

    return ret;
}

SHMEM_DEVICE int32_t shmemi_signal_wait_until_ne(__gm__ int32_t *sig_addr, int32_t cmp_val) {
    int32_t ret;
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while ((ret = *sig_addr) == cmp_val);

    return ret;
}

SHMEM_DEVICE int32_t shmemi_signal_wait_until_gt(__gm__ int32_t *sig_addr, int32_t cmp_val) {
    int32_t ret;
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while ((ret = *sig_addr) <= cmp_val);

    return ret;
}

SHMEM_DEVICE int32_t shmemi_signal_wait_until_ge(__gm__ int32_t *sig_addr, int32_t cmp_val) {
    int32_t ret;
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while ((ret = *sig_addr) < cmp_val);

    return ret;
}

SHMEM_DEVICE int32_t shmemi_signal_wait_until_lt(__gm__ int32_t *sig_addr, int32_t cmp_val) {
    int32_t ret;
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while ((ret = *sig_addr) >= cmp_val);

    return ret;
}

SHMEM_DEVICE int32_t shmemi_signal_wait_until_le(__gm__ int32_t *sig_addr, int32_t cmp_val) {
    int32_t ret;
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);
    } while ((ret = *sig_addr) > cmp_val);

    return ret;
}

SHMEM_DEVICE int32_t shmemi_signal_wait_until(__gm__ int32_t *sig_addr, int cmp, int32_t cmp_val) {
    switch (cmp) {
        case SHMEM_CMP_EQ:
            return shmemi_signal_wait_until_eq(sig_addr, cmp_val);
        case SHMEM_CMP_NE:
            return shmemi_signal_wait_until_ne(sig_addr, cmp_val);
        case SHMEM_CMP_GT:
            return shmemi_signal_wait_until_gt(sig_addr, cmp_val);
        case SHMEM_CMP_GE:
            return shmemi_signal_wait_until_ge(sig_addr, cmp_val);
        case SHMEM_CMP_LT:
            return shmemi_signal_wait_until_lt(sig_addr, cmp_val);
        case SHMEM_CMP_LE:
            return shmemi_signal_wait_until_le(sig_addr, cmp_val);
    }
    return -1;
}

#endif