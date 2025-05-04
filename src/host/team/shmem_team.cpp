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
ShmemiTeam *gShmemTeamPool = nullptr;

inline std::string TeamConfig2String(ShmemiTeam *config)
{
    std::ostringstream oss;
    oss << "[team:" << config->teamIdx;
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

inline void DeviceTeamDestroy(int32_t teamIdx)
{
    // devicePtr Free
    ShmemiTeam *deviceTeamPtr = gState.teamPools[teamIdx];
    if (deviceTeamPtr != nullptr) {
        aclrtFree((void *) deviceTeamPtr);
        gState.teamPools[teamIdx] = nullptr;
    }
}

inline int32_t DeviceTeamUpdate(int teamIdx, ShmemiTeam *hostTeamPtr)
{
    // devicePtr Malloc
    void* teamPtr = nullptr;
    SHMEM_CHECK_RET(aclrtMalloc(&teamPtr, sizeof(ShmemiTeam), ACL_MEM_MALLOC_NORMAL_ONLY));
    auto ret = aclrtMemcpy((ShmemiTeam *)teamPtr, sizeof(ShmemiTeam),
                           hostTeamPtr, sizeof(ShmemiTeam), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != 0) {
        SHM_LOG_ERROR("memcpy device team info failed, ret: " << ret);
        aclrtFree(teamPtr);
        return SHMEM_INNER_ERROR;
    }
    gState.teamPools[teamIdx] = (ShmemiTeam *)teamPtr;
    return SHMEM_SUCCESS;
}

int32_t ShmemiTeamInit(int32_t rank, int32_t size)
{
    /* Initialize SHMEM_TEAM_WORLD */
    gShmemTeamPool = (ShmemiTeam *)calloc(SHMEM_MAX_TEAMS, sizeof(ShmemiTeam));
    if (gShmemTeamPool == nullptr) {
        SHM_LOG_ERROR("malloc host shmem team pool failed.");
        return SHMEM_INNER_ERROR;
    }
    for (int i = 0; i < SHMEM_MAX_TEAMS; i++) {
        gShmemTeamPool[i] = ShmemiTeam{-1, -1, -1, -1, -1};
    }

    ShmemiTeam &shmemTeamWorld = gShmemTeamPool[SHMEM_TEAM_WORLD];
    shmemTeamWorld.teamIdx = SHMEM_TEAM_WORLD;
    shmemTeamWorld.start = 0;
    shmemTeamWorld.stride = 1;
    shmemTeamWorld.size = size;       // TODO state->npes
    shmemTeamWorld.mype = rank;       // TODO state->mype
    gTeamMask |= 1ULL << SHMEM_TEAM_WORLD;
    SHMEM_CHECK_RET(DeviceTeamUpdate(SHMEM_TEAM_WORLD, &shmemTeamWorld));

    /* Initialize TEAM SYNC */
    gState.syncPool = (ShmemiSyncBit *)shmem_malloc(SYNC_POOL_SIZE);
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

    ret = aclrtMalloc((void **) &(gState.syncCounter), SYNC_COUNTERS_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != 0 || gState.syncCounter == nullptr) {
        ShmemiTeamFinalize();
        SHM_LOG_ERROR("malloc sync counter failed.");
        return SHMEM_INNER_ERROR;
    }
    ret = ShmemiMemset((int32_t *) gState.syncCounter, SYNC_COUNTERS_SIZE / sizeof(int32_t), 1);
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

    if (gState.syncCounter != nullptr) {
        (void)aclrtFree(reinterpret_cast<void *>(gState.syncCounter));
        gState.syncCounter = nullptr;
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
        shmem_team_t parentTeam,
        int32_t peStart, int32_t peStride, int32_t peSize,
        shmem_team_t &newTeam)
{
    newTeam = SHMEM_TEAM_INVALID;
    if (!shm::IsValidTeam(parentTeam)) {
        SHM_LOG_ERROR("input parent team is invalid!, team: " << parentTeam);
        return SHMEM_INVALID_PARAM;
    }

    ShmemiTeam myTeam;
    ShmemiTeam *srcTeam = &shm::gShmemTeamPool[parentTeam];

    int32_t globalPE = srcTeam->mype;
    int32_t globalPeStart = srcTeam->start + peStart * srcTeam->stride;
    int32_t globalPeStride = srcTeam->stride * peStride;
    int32_t globalPeEnd = globalPeStart + globalPeStride * (peSize - 1);

    if (peStart < 0 || peStart >= srcTeam->size || peSize <= 0 || peSize > srcTeam->size || peStride < 1) {
        SHM_LOG_ERROR("create team failed, input invalid, peStart:" << peStart << " peSize:" << peSize <<
            " peStride:" << peStride << " parent:" << shm::TeamConfig2String(srcTeam));
        return SHMEM_INVALID_PARAM;
    }

    if (globalPeStart >= shmem_n_pes() || globalPeEnd >= shmem_n_pes()) {
        SHM_LOG_ERROR("create team failed, large than world size, peStart:" << peStart << " peSize:" << peSize <<
            " peStride:" << peStride << " worldSize:" << shmem_n_pes() << " parent:" << shm::TeamConfig2String(srcTeam));
        return SHMEM_INVALID_PARAM;
    }

    myTeam.mype = (globalPE - globalPeStart) / globalPeStride;

    if (globalPE < globalPeStart || (globalPE - globalPeStart)  % globalPeStride || myTeam.mype >= peSize) {
        SHM_LOG_ERROR("create team failed, mype is invalid, peStart:" << peStart << " peSize:" << peSize <<
            " peStride:" << peStride << " mype:" << myTeam.mype << " parent:" << shm::TeamConfig2String(srcTeam));
        return SHMEM_INVALID_PARAM;
    }

    myTeam.start = globalPeStart;
    myTeam.stride = globalPeStride;
    myTeam.size = peSize;

    myTeam.teamIdx = shm::FirstFreeIdxFetch();
    if (myTeam.teamIdx == -1) {
        SHM_LOG_ERROR("create team failed, team num is full!");
        return SHMEM_INNER_ERROR;
    }

    shm::gShmemTeamPool[myTeam.teamIdx] = myTeam;
    if (shm::DeviceTeamUpdate(myTeam.teamIdx, &shm::gShmemTeamPool[myTeam.teamIdx]) != 0) {
        shmem_team_destroy(myTeam.teamIdx);
        SHM_LOG_ERROR("create team failed, malloc device state failed!");
        return SHMEM_INNER_ERROR;
    }
    if (shm::UpdateDeviceState() != 0) {
        shmem_team_destroy(myTeam.teamIdx);
        SHM_LOG_ERROR("create team failed, update state failed!");
        return SHMEM_INNER_ERROR;
    }
    newTeam = myTeam.teamIdx;
    return 0;
}


int32_t shmem_team_translate_pe(
    shmem_team_t srcTeam, int32_t srcPe,
    shmem_team_t destTeam)
{
    if (!shm::IsValidTeam(srcTeam) || !shm::IsValidTeam(destTeam)) {
        return -1;
    }

    ShmemiTeam *srcTeamPtr = &shm::gShmemTeamPool[srcTeam];
    ShmemiTeam *destTeamPtr = &shm::gShmemTeamPool[destTeam];

    if (srcPe > srcTeamPtr->size) return -1;

    int32_t globalPE = srcTeamPtr->start + srcPe * srcTeamPtr->stride;
    int32_t peStart = destTeamPtr->start;
    int32_t peStride = destTeamPtr->stride;
    int32_t peSize = destTeamPtr->size;

    int32_t n = (globalPE - peStart) / peStride;
    if (globalPE < peStart || (globalPE - peStart) % peStride || n >= peSize)
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
    ShmemiBarrierOnStream(tid, nullptr);
}

void shmem_barrier_all() {
    shmem_barrier(SHMEM_TEAM_WORLD);
}