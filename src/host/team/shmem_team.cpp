/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "shmemi_device_intf.h"
using namespace std;

namespace shm {
uint64_t g_team_mask = 0;
shmemi_team_t *g_shmem_team_pool = nullptr;

inline std::string team_config2string(shmemi_team_t *config)
{
    std::ostringstream oss;
    oss << "[team:" << config->team_idx;
    oss << ",npes:" << config->size;
    oss << ",mype:" << config->mype;
    oss << ",start:" << config->start;
    oss << ",stride:" << config->stride;
    oss << "]";
    return oss.str();
}

inline bool is_valid_team(shmem_team_t &team)
{
    return (g_state.is_shmem_initialized && g_shmem_team_pool != nullptr &&
        team >= 0 && team < SHMEM_MAX_TEAMS && (g_team_mask >> team & 1));
}

inline void device_team_destroy(int32_t team_idx)
{
    // device_ptr Free
    shmemi_team_t *device_team_ptr = g_state.team_pools[team_idx];
    if (device_team_ptr != nullptr) {
        if (aclrtFree((void *) device_team_ptr) != ACL_SUCCESS) {
            SHM_LOG_ERROR("aclrtFree for device_team_ptr failed for team " << team_idx);
        }
        g_state.team_pools[team_idx] = nullptr;
    }
}

inline int32_t device_team_update(int team_idx, shmemi_team_t *host_team_ptr)
{
    // device_ptr Malloc
    void* team_ptr = nullptr;
    SHMEM_CHECK_RET(aclrtMalloc(&team_ptr, sizeof(shmemi_team_t), ACL_MEM_MALLOC_NORMAL_ONLY));
    auto ret = aclrtMemcpy((shmemi_team_t *)team_ptr, sizeof(shmemi_team_t),
                           host_team_ptr, sizeof(shmemi_team_t), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != 0) {
        SHM_LOG_ERROR("memcpy device team info failed, ret: " << ret);
        aclrtFree(team_ptr);
        return SHMEM_INNER_ERROR;
    }
    g_state.team_pools[team_idx] = (shmemi_team_t *)team_ptr;
    return SHMEM_SUCCESS;
}

int32_t shmemi_team_init(int32_t rank, int32_t size)
{
    /* Initialize SHMEM_TEAM_WORLD */
    g_shmem_team_pool = (shmemi_team_t *)calloc(SHMEM_MAX_TEAMS, sizeof(shmemi_team_t));
    if (g_shmem_team_pool == nullptr) {
        SHM_LOG_ERROR("malloc host shmem team pool failed.");
        return SHMEM_INNER_ERROR;
    }
    for (int i = 0; i < SHMEM_MAX_TEAMS; i++) {
        g_shmem_team_pool[i] = shmemi_team_t{-1, -1, -1, -1, -1};
    }

    shmemi_team_t &shmem_team_world = g_shmem_team_pool[SHMEM_TEAM_WORLD];
    shmem_team_world.team_idx = SHMEM_TEAM_WORLD;
    shmem_team_world.start = 0;
    shmem_team_world.stride = 1;
    shmem_team_world.size = size;       // TODO state->npes
    shmem_team_world.mype = rank;       // TODO state->mype
    g_team_mask |= 1ULL << SHMEM_TEAM_WORLD;
    SHMEM_CHECK_RET(device_team_update(SHMEM_TEAM_WORLD, &shmem_team_world));

    /* Initialize TEAM SYNC */
    g_state.sync_pool = (uint64_t)shmem_malloc(SYNC_POOL_SIZE);
    if (g_state.sync_pool == 0) {
        shmemi_team_finalize();
        SHM_LOG_ERROR("malloc sync pool failed.");
        return SHMEM_INNER_ERROR;
    }
    auto ret = aclrtMemset((void *) g_state.sync_pool, SYNC_POOL_SIZE, 0, SYNC_POOL_SIZE);
    if (ret != 0) {
        shmemi_team_finalize();
        SHM_LOG_ERROR("memset sync pool failed.");
        return SHMEM_INNER_ERROR;
    }

    ret = aclrtMalloc((void **) &(g_state.sync_counter), SYNC_COUNTERS_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != 0 || g_state.sync_counter == 0) {
        shmemi_team_finalize();
        SHM_LOG_ERROR("malloc sync counter failed.");
        return SHMEM_INNER_ERROR;
    }
    ret = aclrtMemset((void *) g_state.sync_counter, SYNC_COUNTERS_SIZE, 0, SYNC_COUNTERS_SIZE);
    if (ret != 0) {
        shmemi_team_finalize();
        SHM_LOG_ERROR("memset sync counter failed.");
        return SHMEM_INNER_ERROR;
    }

    ret = aclrtMalloc((void **) &(g_state.core_sync_pool), SHMEM_CORE_SYNC_POOL_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != 0 || g_state.core_sync_pool == 0) {
        shmemi_team_finalize();
        SHM_LOG_ERROR("malloc core sync pool failed.");
        return SHMEM_INNER_ERROR;
    }
    ret = aclrtMemset((void *) g_state.core_sync_pool, SHMEM_CORE_SYNC_POOL_SIZE, 0, SHMEM_CORE_SYNC_POOL_SIZE);
    if (ret != 0) {
        shmemi_team_finalize();
        SHM_LOG_ERROR("memset core sync pool failed.");
        return SHMEM_INNER_ERROR;
    }

    ret = aclrtMalloc((void **) &(g_state.core_sync_counter), SHMEM_CORE_SYNC_COUNTER_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != 0 || g_state.core_sync_counter == 0) {
        shmemi_team_finalize();
        SHM_LOG_ERROR("malloc core sync counter failed.");
        return SHMEM_INNER_ERROR;
    }
    ret = aclrtMemset((void *) g_state.core_sync_counter, SHMEM_CORE_SYNC_COUNTER_SIZE, 0, SHMEM_CORE_SYNC_COUNTER_SIZE);
    if (ret != 0) {
        shmemi_team_finalize();
        SHM_LOG_ERROR("memset core sync counter failed.");
        return SHMEM_INNER_ERROR;
    }
    return 0;
}


int32_t first_free_idx_fetch()
{
    int32_t shmem_max_teams = SHMEM_MAX_TEAMS;
    for (int32_t i = 0; i < shmem_max_teams; i++) {
        if (!((g_team_mask >> i) & 1)) {
            g_team_mask |= 1ULL << i;
            return i;
        }
    }
    return -1;
}

int32_t shmemi_team_finalize()
{
    /* Destroy all undestroyed teams */
    int32_t shmem_max_teams = SHMEM_MAX_TEAMS;
    for (int32_t i = 0; i < shmem_max_teams; i++) {
        if (is_valid_team(i)) shmem_team_destroy(i);
    }

    if (g_state.sync_counter != 0) {
        aclrtFree(reinterpret_cast<void *>(g_state.sync_counter));
        g_state.sync_counter = 0;
    }
    if (g_state.sync_pool != 0) {
        shmem_free(reinterpret_cast<void *>(g_state.sync_pool));
        g_state.sync_pool = 0;
    }
    if (g_state.core_sync_counter != 0) {
        aclrtFree(reinterpret_cast<void *>(g_state.core_sync_counter));
        g_state.core_sync_counter = 0;
    }    
    if (g_state.core_sync_pool != 0) {
        aclrtFree(reinterpret_cast<void *>(g_state.core_sync_pool));
        g_state.core_sync_pool = 0;
    }
    if (g_shmem_team_pool != nullptr) {
        free(g_shmem_team_pool);
        g_shmem_team_pool = nullptr;
    }
    return 0;
}

} // namespace shm

int32_t shmem_team_split_strided(
        shmem_team_t parent_team,
        int32_t pe_start, int32_t pe_stride, int32_t pe_size,
        shmem_team_t *new_team)
{
    if (new_team == nullptr) {
        SHM_LOG_ERROR("output team is null.");
        return SHMEM_INVALID_PARAM;
    }

    *new_team = SHMEM_TEAM_INVALID;
    if (!shm::is_valid_team(parent_team)) {
        SHM_LOG_ERROR("input parent team is invalid!, team: " << parent_team);
        return SHMEM_INVALID_PARAM;
    }

    shmemi_team_t my_team;
    shmemi_team_t *src_team = &shm::g_shmem_team_pool[parent_team];

    int32_t global_pe = src_team->mype;
    int32_t global_pe_start = src_team->start + pe_start * src_team->stride;
    int32_t global_pe_stride = src_team->stride * pe_stride;
    int32_t global_pe_end = global_pe_start + global_pe_stride * (pe_size - 1);

    if (pe_start < 0 || pe_start >= src_team->size || pe_size <= 0 || pe_size > src_team->size || pe_stride < 1) {
        SHM_LOG_ERROR("create team failed, input invalid, pe_start:" << pe_start << " pe_size:" << pe_size <<
            " pe_stride:" << pe_stride << " parent:" << shm::team_config2string(src_team));
        return SHMEM_INVALID_PARAM;
    }

    if (global_pe_start >= shmem_n_pes() || global_pe_end >= shmem_n_pes()) {
        SHM_LOG_ERROR("create team failed, large than world size, pe_start:" << pe_start << " pe_size:" << pe_size <<
            " pe_stride:" << pe_stride << " world_size:" << shmem_n_pes() << " parent:" << shm::team_config2string(src_team));
        return SHMEM_INVALID_PARAM;
    }

    my_team.mype = (global_pe - global_pe_start) / global_pe_stride;

    if (global_pe < global_pe_start || (global_pe - global_pe_start)  % global_pe_stride || my_team.mype >= pe_size) {
        SHM_LOG_ERROR("create team failed, mype is invalid, pe_start:" << pe_start << " pe_size:" << pe_size <<
            " pe_stride:" << pe_stride << " mype:" << my_team.mype << " parent:" << shm::team_config2string(src_team));
        return SHMEM_INVALID_PARAM;
    }

    my_team.start = global_pe_start;
    my_team.stride = global_pe_stride;
    my_team.size = pe_size;

    my_team.team_idx = shm::first_free_idx_fetch();
    if (my_team.team_idx == -1) {
        SHM_LOG_ERROR("create team failed, team num is full!");
        return SHMEM_INNER_ERROR;
    }

    shm::g_shmem_team_pool[my_team.team_idx] = my_team;
    if (shm::device_team_update(my_team.team_idx, &shm::g_shmem_team_pool[my_team.team_idx]) != 0) {
        shmem_team_destroy(my_team.team_idx);
        SHM_LOG_ERROR("create team failed, malloc device state failed!");
        return SHMEM_INNER_ERROR;
    }
    if (shm::update_device_state() != 0) {
        shmem_team_destroy(my_team.team_idx);
        SHM_LOG_ERROR("create team failed, update state failed!");
        return SHMEM_INNER_ERROR;
    }
    *new_team = my_team.team_idx;
    return 0;
}


int32_t shmem_team_translate_pe(
    shmem_team_t src_team, int32_t src_pe,
    shmem_team_t dest_team)
{
    if (!shm::is_valid_team(src_team) || !shm::is_valid_team(dest_team)) {
        return -1;
    }

    shmemi_team_t *src_team_ptr = &shm::g_shmem_team_pool[src_team];
    shmemi_team_t *dest_team_ptr = &shm::g_shmem_team_pool[dest_team];

    if (src_pe > src_team_ptr->size) return -1;

    int32_t global_pe = src_team_ptr->start + src_pe * src_team_ptr->stride;
    int32_t pe_start = dest_team_ptr->start;
    int32_t pe_stride = dest_team_ptr->stride;
    int32_t pe_size = dest_team_ptr->size;

    int32_t n = (global_pe - pe_start) / pe_stride;
    if (global_pe < pe_start || (global_pe - pe_start) % pe_stride || n >= pe_size)
        return -1;
    
    return n;
}


void shmem_team_destroy(shmem_team_t team)
{
    if (!shm::is_valid_team(team)) {
        SHM_LOG_WARN("input team is invalid!, team: " << team);
        return;
    }

    shm::device_team_destroy(team);
    shm::g_team_mask ^= 1ULL << team;
    if (shm::update_device_state() != SHMEM_SUCCESS) {
        SHM_LOG_WARN("update state failed when destroy team!");
    }
}

int32_t shmem_my_pe()
{
    return shm::g_state.mype;
}


int32_t shmem_n_pes()
{
    return shm::g_state.npes;
}


int32_t shmem_team_my_pe(shmem_team_t team)
{
    if (shm::is_valid_team(team)) {
        return shm::g_shmem_team_pool[team].mype;
    } else {
        return -1;
    }
}


int32_t shmem_team_n_pes(shmem_team_t team)
{
    if (shm::is_valid_team(team)) {
        return shm::g_shmem_team_pool[team].size;
    } else {
        return -1;
    }
}