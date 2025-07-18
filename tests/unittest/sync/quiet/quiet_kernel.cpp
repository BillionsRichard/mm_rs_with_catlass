/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "shmem_api.h"

extern "C" SHMEM_GLOBAL void quiet(uint64_t config, GM_ADDR addr, GM_ADDR dev, int32_t rank_id, int32_t rank_size) {
    shmemx_set_ffts_config(config);

    shmem_put_int32_mem_nbi((__gm__ int32_t *)addr, (__gm__ int32_t *)dev, rank_size, rank_id);
    shmemi_store((__gm__ int32_t *)dev, rank_id + 11);
    shmem_put_int32_mem_nbi((__gm__ int32_t *)addr, (__gm__ int32_t *)dev, rank_size, rank_id);
    shmem_quiet();
    shmemi_store((__gm__ int32_t *)dev, rank_id + 12);
    shmem_put_int32_mem_nbi((__gm__ int32_t *)addr, (__gm__ int32_t *)dev, rank_size, rank_id);
}

void quiet_do(void* stream, uint64_t config, uint8_t *addr, uint8_t *dev, int32_t rank_id, int32_t rank_size) {
    quiet<<<1, nullptr, stream>>>(config, addr, dev, rank_id, rank_size);
}
