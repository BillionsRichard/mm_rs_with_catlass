#ifndef SHMEM_DEVICE_TEAM_H
#define SHMEM_DEVICE_TEAM_H

#include "host_device/shmem_types.h"
#include "internal/host_device/shmemi_types.h"

SHMEM_DEVICE int shmem_my_pe(void)
{
    return ShmemiGetState()->teamPools[SHMEM_TEAM_WORLD]->mype;
}

SHMEM_DEVICE int shmem_n_pes(void)
{
    return ShmemiGetState()->teamPools[SHMEM_TEAM_WORLD]->size;
}

SHMEM_DEVICE int shmem_team_my_pe(shmem_team_t team)
{
    if (team == SHMEM_TEAM_INVALID)
        return -1;
    else {
        ShmemiTeam *srcTeamPtr = ShmemiGetState()->teamPools[team];
        return (srcTeamPtr != nullptr ? srcTeamPtr->mype : -1);
    }
}

SHMEM_DEVICE int shmem_team_n_pes(shmem_team_t team)
{
    if (team == SHMEM_TEAM_INVALID)
        return -1;
    else {
        ShmemiTeam *srcTeamPtr = ShmemiGetState()->teamPools[team];
        return (srcTeamPtr != nullptr ? srcTeamPtr->size : -1);
    }
}

SHMEM_DEVICE int shmem_team_translate_pe(shmem_team_t srcTeam, int srcPe, shmem_team_t destTeam)
{
    if (srcTeam == SHMEM_TEAM_INVALID || destTeam == SHMEM_TEAM_INVALID) return -1;
    __gm__ ShmemiDeviceHostState *deviceState = ShmemiGetState();

    ShmemiTeam *srcTeamPtr = deviceState->teamPools[srcTeam];
    ShmemiTeam *destTeamPtr = deviceState->teamPools[destTeam];

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