/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "acl/acl.h"
#include "kernel_operator.h"

#include "shmem_api.h"

// kernels
template<typename T>
SHMEM_GLOBAL void k_memset(GM_ADDR array, int32_t len, T val, int32_t count) {
    if (array == 0) {
        return;
    }
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