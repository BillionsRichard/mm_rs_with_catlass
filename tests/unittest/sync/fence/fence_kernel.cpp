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

extern "C" SHMEM_GLOBAL void fence_order(uint64_t config, GM_ADDR addr, int rank_id, int rank_size) {
    shmemx_set_ffts_config(config);
    __gm__ uint64_t *base = reinterpret_cast<__gm__ uint64_t*>(addr);
    shmem_barrier_all();
    if (rank_id == 0) {
        shmemi_store<uint64_t>(base, 0xAA);
        shmemi_store<uint64_t>(base + 1, 0xBB);
        shmemi_store<uint64_t>(base + 2, 0xCC);
        shmem_fence();
    }

    if (rank_id == 1) {
        uint64_t seen_c;
        __gm__ uint64_t *remote = shmemi_ptr(base, 0);
        do {
            seen_c = shmemi_load<uint64_t>(remote + 2);
        } while (seen_c != 0xCC);

        uint64_t seen_b = shmemi_load<uint64_t>(remote + 1);
        uint64_t seen_a = shmemi_load<uint64_t>(remote);

        shmemi_store<uint64_t>(base + 3, seen_c);
        shmemi_store<uint64_t>(base + 4, seen_b);
        shmemi_store<uint64_t>(base + 5, seen_a);
    }

    shmem_barrier_all();
}

void fence_order_do(void* stream, uint64_t config, uint8_t *addr, int32_t rank_id, int32_t n_ranks) {
    fence_order<<<1, nullptr, stream>>>(config, addr, rank_id, n_ranks);
}
