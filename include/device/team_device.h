#ifndef _DEVICE_TEAM_
#define _DEVICE_TEAM_

#include "kernel_operator.h"
#include "low_level_api/smem_shm_aicore_base_api.h"

#include "shmem_device_api.h"

__aicore__ inline int ShmemMype(void)
{
    __gm__ void* addrGM = smem_shm_get_extra_context_addr();
    __gm__ ShmemDeviceHostState *deviceState = (__gm__ ShmemDeviceHostState *)addrGM;
    return deviceState->teamPools[SHMEM_TEAM_WORLD]->mype;
}

__aicore__ inline int ShmemNpes(void)
{
    __gm__ void* addrGM = smem_shm_get_extra_context_addr();
    __gm__ ShmemDeviceHostState *deviceState = (__gm__ ShmemDeviceHostState *)addrGM;
    return deviceState->teamPools[SHMEM_TEAM_WORLD]->size;
}

__aicore__ inline int ShmemTeamMype(ShmemTeam_t team)
{
    if (team == SHMEM_TEAM_INVALID)
        return -1;
    else {
        __gm__ void* addrGM = smem_shm_get_extra_context_addr();
        __gm__ ShmemDeviceHostState *deviceState = (__gm__ ShmemDeviceHostState *)addrGM;
        return deviceState->teamPools[team]->mype;
    }
}

__aicore__ inline int ShmemTeamNpes(ShmemTeam_t team)
{
    if (team == SHMEM_TEAM_INVALID)
        return -1;
    else {
        __gm__ void* addrGM = smem_shm_get_extra_context_addr();
        __gm__ ShmemDeviceHostState *deviceState = (__gm__ ShmemDeviceHostState *)addrGM;
        return deviceState->teamPools[team]->size;
    }
}

__aicore__ inline int ShmemTeamTranslatePE(ShmemTeam_t srcTeam, int srcPe, ShmemTeam_t destTeam)
{
    if (srcTeam == SHMEM_TEAM_INVALID || destTeam == SHMEM_TEAM_INVALID) return -1;
    __gm__ void* addrGM = smem_shm_get_extra_context_addr();
    __gm__ ShmemDeviceHostState *deviceState = (__gm__ ShmemDeviceHostState *)addrGM;

    ShmemTeam *srcTeamPtr = deviceState->teamPools[srcTeam];
    ShmemTeam *destTeamPtr = deviceState->teamPools[destTeam];

    if (srcPe > srcTeamPtr->size) return -1;

    int globalPE = srcTeamPtr->start + srcPe * srcTeamPtr->stride;
    int peStart = destTeamPtr->start;
    int peStride = destTeamPtr->stride;
    int peSize = destTeamPtr->size;

    int n = (globalPE - peStart) / peStride;
    if (globalPE < peStart || (globalPE - peStart) % peStride || n >= peSize)
        return -1;
    
    return n;
}

#endif