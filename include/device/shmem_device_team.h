#ifndef SHMEM_DEVICE_TEAM_H
#define SHMEM_DEVICE_TEAM_H

#include "kernel_operator.h"
#include "host_device/shmem_types.h"
#include "internal/host_device/shmemi_types.h"

SHMEM_DEVICE int ShmemMype(void)
{
    return ShmemiGetState()->teamPools[SHMEM_TEAM_WORLD]->mype;
}

SHMEM_DEVICE int ShmemNpes(void)
{
    return ShmemiGetState()->teamPools[SHMEM_TEAM_WORLD]->size;
}

SHMEM_DEVICE int ShmemTeamMype(ShmemTeam team)
{
    if (team == SHMEM_TEAM_INVALID)
        return -1;
    else {
        return ShmemiGetState()->teamPools[team]->mype;
    }
}

SHMEM_DEVICE int ShmemTeamNpes(ShmemTeam team)
{
    if (team == SHMEM_TEAM_INVALID)
        return -1;
    else {
        return ShmemiGetState()->teamPools[team]->size;
    }
}

SHMEM_DEVICE int ShmemTeamTranslatePE(ShmemTeam srcTeam, int srcPe, ShmemTeam destTeam)
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