#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>

using namespace std;

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "shmemi_device_intf.h"

namespace shm {
uint64_t gTeamMask = 0;
shmemi_team_t *gShmemTeamPool = nullptr;

inline std::string TeamConfig2String(shmemi_team_t *config)
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

inline bool IsValidTeam(shmem_team_t &team)
{
    return (gState.isShmemInitialized && gShmemTeamPool != nullptr &&
        team >= 0 && team < SHMEM_MAX_TEAMS && (gTeamMask >> team & 1));
}

inline void DeviceTeamDestroy(int32_t team_idx)
{
    // devicePtr Free
    shmemi_team_t *deviceTeamPtr = gState.team_pools[team_idx];
    if (deviceTeamPtr != nullptr) {
        aclrtFree((void *) deviceTeamPtr);
        gState.team_pools[team_idx] = nullptr;
    }
}

inline int32_t DeviceTeamUpdate(int team_idx, shmemi_team_t *hostTeamPtr)
{
    // devicePtr Malloc
    void* teamPtr = nullptr;
    SHMEM_CHECK_RET(aclrtMalloc(&teamPtr, sizeof(shmemi_team_t), ACL_MEM_MALLOC_NORMAL_ONLY));
    auto ret = aclrtMemcpy((shmemi_team_t *)teamPtr, sizeof(shmemi_team_t),
                           hostTeamPtr, sizeof(shmemi_team_t), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != 0) {
        SHM_LOG_ERROR("memcpy device team info failed, ret: " << ret);
        aclrtFree(teamPtr);
        return SHMEM_INNER_ERROR;
    }
    gState.team_pools[team_idx] = (shmemi_team_t *)teamPtr;
    return SHMEM_SUCCESS;
}

int32_t ShmemiTeamInit(int32_t rank, int32_t size)
{
    /* Initialize SHMEM_TEAM_WORLD */
    gShmemTeamPool = (shmemi_team_t *)calloc(SHMEM_MAX_TEAMS, sizeof(shmemi_team_t));
    if (gShmemTeamPool == nullptr) {
        SHM_LOG_ERROR("malloc host shmem team pool failed.");
        return SHMEM_INNER_ERROR;
    }
    for (int i = 0; i < SHMEM_MAX_TEAMS; i++) {
        gShmemTeamPool[i] = shmemi_team_t{-1, -1, -1, -1, -1};
    }

    shmemi_team_t &shmemTeamWorld = gShmemTeamPool[SHMEM_TEAM_WORLD];
    shmemTeamWorld.team_idx = SHMEM_TEAM_WORLD;
    shmemTeamWorld.start = 0;
    shmemTeamWorld.stride = 1;
    shmemTeamWorld.size = size;       // TODO state->npes
    shmemTeamWorld.mype = rank;       // TODO state->mype
    gTeamMask |= 1ULL << SHMEM_TEAM_WORLD;
    SHMEM_CHECK_RET(DeviceTeamUpdate(SHMEM_TEAM_WORLD, &shmemTeamWorld));

    /* Initialize TEAM SYNC */
    gState.syncPool = (shmemi_sync_bit *)shmem_malloc(SYNC_POOL_SIZE);
    if (gState.syncPool == nullptr) {
        ShmemiTeamFinalize();
        SHM_LOG_ERROR("malloc sync pool failed.");
        return SHMEM_INNER_ERROR;
    }
    auto ret = aclrtMemset((void *) gState.syncPool, SYNC_POOL_SIZE, 0, SYNC_POOL_SIZE);
    if (ret != 0) {
        ShmemiTeamFinalize();
        SHM_LOG_ERROR("memset sync pool failed.");
        return SHMEM_INNER_ERROR;
    }

    ret = aclrtMalloc((void **) &(gState.sync_counter), SYNC_COUNTERS_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != 0 || gState.sync_counter == nullptr) {
        ShmemiTeamFinalize();
        SHM_LOG_ERROR("malloc sync counter failed.");
        return SHMEM_INNER_ERROR;
    }
    ret = shmemi_memset((int32_t *) gState.sync_counter, SYNC_COUNTERS_SIZE / sizeof(int32_t), 1);
    if (ret != 0) {
        ShmemiTeamFinalize();
        SHM_LOG_ERROR("memset sync counter failed.");
        return SHMEM_INNER_ERROR;
    }
    return 0;
}


int32_t FirstFreeIdxFetch()
{
    int32_t shmemMaxTeams = SHMEM_MAX_TEAMS;
    for (int32_t i = 0; i < shmemMaxTeams; i++) {
        if (!((gTeamMask >> i) & 1)) {
            gTeamMask |= 1ULL << i;
            return i;
        }
    }
    return -1;
}

int32_t ShmemiTeamFinalize()
{
    /* Destroy all undestroyed teams*/
    int32_t shmemMaxTeams = SHMEM_MAX_TEAMS;
    for (int32_t i = 0; i < shmemMaxTeams; i++) {
        if (IsValidTeam(i)) shmem_team_destroy(i);
    }

    if (gState.sync_counter != nullptr) {
        (void)aclrtFree(reinterpret_cast<void *>(gState.sync_counter));
        gState.sync_counter = nullptr;
    }
    if (gState.syncPool != nullptr) {
        shmem_free(reinterpret_cast<void *>(gState.syncPool));
        gState.syncPool = nullptr;
    }
    if (gShmemTeamPool != nullptr) {
        free(gShmemTeamPool);
        gShmemTeamPool = nullptr;
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
    if (!shm::IsValidTeam(parent_team)) {
        SHM_LOG_ERROR("input parent team is invalid!, team: " << parent_team);
        return SHMEM_INVALID_PARAM;
    }

    shmemi_team_t myTeam;
    shmemi_team_t *src_team = &shm::gShmemTeamPool[parent_team];

    int32_t global_pe = src_team->mype;
    int32_t globalPeStart = src_team->start + pe_start * src_team->stride;
    int32_t globalPeStride = src_team->stride * pe_stride;
    int32_t globalPeEnd = globalPeStart + globalPeStride * (pe_size - 1);

    if (pe_start < 0 || pe_start >= src_team->size || pe_size <= 0 || pe_size > src_team->size || pe_stride < 1) {
        SHM_LOG_ERROR("create team failed, input invalid, pe_start:" << pe_start << " pe_size:" << pe_size <<
            " pe_stride:" << pe_stride << " parent:" << shm::TeamConfig2String(src_team));
        return SHMEM_INVALID_PARAM;
    }

    if (globalPeStart >= shmem_n_pes() || globalPeEnd >= shmem_n_pes()) {
        SHM_LOG_ERROR("create team failed, large than world size, pe_start:" << pe_start << " pe_size:" << pe_size <<
            " pe_stride:" << pe_stride << " worldSize:" << shmem_n_pes() << " parent:" << shm::TeamConfig2String(src_team));
        return SHMEM_INVALID_PARAM;
    }

    myTeam.mype = (global_pe - globalPeStart) / globalPeStride;

    if (global_pe < globalPeStart || (global_pe - globalPeStart)  % globalPeStride || myTeam.mype >= pe_size) {
        SHM_LOG_ERROR("create team failed, mype is invalid, pe_start:" << pe_start << " pe_size:" << pe_size <<
            " pe_stride:" << pe_stride << " mype:" << myTeam.mype << " parent:" << shm::TeamConfig2String(src_team));
        return SHMEM_INVALID_PARAM;
    }

    myTeam.start = globalPeStart;
    myTeam.stride = globalPeStride;
    myTeam.size = pe_size;

    myTeam.team_idx = shm::FirstFreeIdxFetch();
    if (myTeam.team_idx == -1) {
        SHM_LOG_ERROR("create team failed, team num is full!");
        return SHMEM_INNER_ERROR;
    }

    shm::gShmemTeamPool[myTeam.team_idx] = myTeam;
    if (shm::DeviceTeamUpdate(myTeam.team_idx, &shm::gShmemTeamPool[myTeam.team_idx]) != 0) {
        shmem_team_destroy(myTeam.team_idx);
        SHM_LOG_ERROR("create team failed, malloc device state failed!");
        return SHMEM_INNER_ERROR;
    }
    if (shm::UpdateDeviceState() != 0) {
        shmem_team_destroy(myTeam.team_idx);
        SHM_LOG_ERROR("create team failed, update state failed!");
        return SHMEM_INNER_ERROR;
    }
    *new_team = myTeam.team_idx;
    return 0;
}


int32_t shmem_team_translate_pe(
    shmem_team_t src_team, int32_t src_pe,
    shmem_team_t dest_team)
{
    if (!shm::IsValidTeam(src_team) || !shm::IsValidTeam(dest_team)) {
        return -1;
    }

    shmemi_team_t *src_team_ptr = &shm::gShmemTeamPool[src_team];
    shmemi_team_t *dest_team_ptr = &shm::gShmemTeamPool[dest_team];

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
    if (!shm::IsValidTeam(team)) {
        SHM_LOG_WARN("input team is invalid!, team: " << team);
        return;
    }

    shm::DeviceTeamDestroy(team);
    shm::gTeamMask ^= 1ULL << team;
    if (shm::UpdateDeviceState() != SHMEM_SUCCESS) {
        SHM_LOG_WARN("update state failed when destroy team!");
    }
}

int32_t shmem_my_pe()
{
    return shm::gState.mype;
}


int32_t shmem_n_pes()
{
    return shm::gState.npes;
}


int32_t shmem_team_my_pe(shmem_team_t team)
{
    if (shm::IsValidTeam(team)) {
        return shm::gShmemTeamPool[team].mype;
    } else {
        return -1;
    }
}


int32_t shmem_team_n_pes(shmem_team_t team)
{
    if (shm::IsValidTeam(team)) {
        return shm::gShmemTeamPool[team].size;
    } else {
        return -1;
    }
}

void shmem_barrier(shmem_team_t tid) {
    // using default stream to do barrier
    shmemi_barrier_on_stream(tid, nullptr);
}

void shmem_barrier_all() {
    shmem_barrier(SHMEM_TEAM_WORLD);
}

void shmem_barrier_on_stream(shmem_team_t tid, aclrtStream stream)
{
    shmemi_barrier_on_stream(tid, stream);
}

void shmem_barrier_all_on_stream(aclrtStream stream)
{
    shmemi_barrier_on_stream(SHMEM_TEAM_WORLD, stream);
}