/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_TYPES_H
#define SHMEMI_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#define SHMEM_MAX_RANKS 2048
#define SHMEM_MAX_TEAMS 32

/* arch related */
#define SCALAR_DATA_CACHELINE_SIZE 64
#define L2_CACHELINE_SIZE 512

#define SHMEM_PAGE_SIZE (1024UL * 1024UL * 2)

#define ALIGH_TO(size, page) (((size) + (page) - 1) / (page) * (page))

/* synchonization related */ 
#define SHMEMI_SYNCBIT_SIZE SCALAR_DATA_CACHELINE_SIZE

// npu level sync
#define SYNC_ARRAY_SIZE (SHMEMI_SYNCBIT_SIZE * SHMEM_MAX_RANKS)
#define SYNC_COUNTER_SIZE SHMEMI_SYNCBIT_SIZE
#define SYNC_POOL_SIZE (SYNC_ARRAY_SIZE * SHMEM_MAX_TEAMS)
#define SYNC_COUNTERS_SIZE (SYNC_COUNTER_SIZE * SHMEM_MAX_TEAMS)
#define SHMEM_BARRIER_TG_DISSEM_KVAL 8

// core level sync
#define SHMEM_MAX_AIV_PER_NPU 48
#define SHMEM_LOG_MAX_AIV_PER_NPU 6     // ceil(log_{2}^{48}) = 6
#define SHMEM_CORE_SYNC_POOL_SIZE (SHMEM_MAX_AIV_PER_NPU * SHMEM_LOG_MAX_AIV_PER_NPU * SHMEMI_SYNCBIT_SIZE)
#define SHMEM_CORE_SYNC_COUNTER_SIZE SHMEMI_SYNCBIT_SIZE

// Total extra
#define SHMEM_EXTRA_SIZE_UNALIGHED SYNC_POOL_SIZE
#define SHMEM_EXTRA_SIZE ALIGH_TO(SHMEM_EXTRA_SIZE_UNALIGHED, SHMEM_PAGE_SIZE)

// synchronization
typedef int32_t shmemi_sync_bit[SHMEMI_SYNCBIT_SIZE / sizeof(int32_t)];

// Team
typedef struct {
    int mype;           // team view, [0, size]
    int start;          // global view, [0, npes]
    int stride;         // global view, [1, npes - 1]
    int size;           // team view
    int team_idx;
} shmemi_team_t;

// mte_config
typedef struct {
    int64_t shmem_ub;        // __ubuf__ Ptr, Shmem memcpy needed.
    uint32_t ub_size;        // UB's Size, in Bytes.
    uint32_t event_id;       // TEventID, for Shmem memcpy sync.
} shmemi_mte_config_t;

// state
typedef struct {
    int version;
    int mype;
    int npes;
    void *heap_base;
    void *p2p_heap_base[SHMEM_MAX_RANKS];
    void *sdma_heap_base[SHMEM_MAX_RANKS];
    void *roce_heap_base[SHMEM_MAX_RANKS];
    size_t heap_size;

    shmemi_team_t *team_pools[SHMEM_MAX_TEAMS];
    
    // Using shmemi_sync_bit instead of basic types to shmemi_store flag, avoiding concurrent write due to cacheline sharing.
    // Refer to shmemi_barrier.h for more details.
    // These members are 'shmemi_sync_bit *' types actully, but are defined as 'uint64_t' due to compiler restriction.
    uint64_t sync_pool;
    uint64_t sync_counter;
    uint64_t core_sync_pool;
    uint64_t core_sync_counter;

    bool is_shmem_initialized;
    bool is_shmem_created;

    shmemi_mte_config_t mte_config;
} shmemi_device_host_state_t;

#ifdef __cplusplus
}
#endif

#endif