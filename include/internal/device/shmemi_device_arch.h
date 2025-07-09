/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_DEVICE_ARCH_H
#define SHMEMI_DEVICE_ARCH_H

#include "device/shmem_device_def.h"

SHMEM_DEVICE void dcci_cacheline(__gm__ uint8_t * addr) {
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(addr);

    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

SHMEM_DEVICE void dcci_entire_cache() {
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    
    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

SHMEM_DEVICE void dcci_atomic() {
    using namespace AscendC;
    GlobalTensor<uint8_t> global;

    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_ATOMIC>(global);
    __asm__ __volatile__("");
}

SHMEM_DEVICE void dsb_all() {
    using namespace AscendC;
    
    DataSyncBarrier<MemDsbT::ALL>();
}

#endif