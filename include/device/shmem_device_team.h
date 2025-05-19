#ifndef SHMEM_DEVICE_TEAM_H
#define SHMEM_DEVICE_TEAM_H

#include "host_device/shmem_types.h"
#include "internal/host_device/shmemi_types.h"

/**
 * @brief Returns the PE number of the local PE
 *
 * @return Integer between 0 and npes - 1
 */
SHMEM_DEVICE int shmem_my_pe(void)
{
    return ShmemiGetState()->teamPools[SHMEM_TEAM_WORLD]->mype;
}

/**
 * @brief Returns the number of PEs running in the program.
 *
 * @return Number of PEs in the program.
 */
SHMEM_DEVICE int shmem_n_pes(void)
{
    return ShmemiGetState()->teamPools[SHMEM_TEAM_WORLD]->size;
}

/**
 * @brief Returns the number of the calling PE in the specified team.
 * 
 * @param team              [in] A team handle.
 *
 * @return The number of the calling PE within the specified team. 
 *         If the team handle is NVSHMEM_TEAM_INVALID, returns -1.
 */
SHMEM_DEVICE int shmem_team_my_pe(shmem_team_t team)
{
    if (team == SHMEM_TEAM_INVALID)
        return -1;
    else {
        ShmemiTeam *srcTeamPtr = ShmemiGetState()->teamPools[team];
        return (srcTeamPtr != nullptr ? srcTeamPtr->mype : -1);
    }
}

/**
 * @brief Returns the number of PEs in the team.
 * 
 * @param team              [in] A team handle.
 *
 * @return The number of PEs in the specified team. 
 *         If the team handle is NVSHMEM_TEAM_INVALID, returns -1.
 */
SHMEM_DEVICE int shmem_team_n_pes(shmem_team_t team)
{
    if (team == SHMEM_TEAM_INVALID)
        return -1;
    else {
        ShmemiTeam *srcTeamPtr = ShmemiGetState()->teamPools[team];
        return (srcTeamPtr != nullptr ? srcTeamPtr->size : -1);
    }
}

/**
 * @brief Translate a given PE number in one team into the corresponding PE number in another team.
 * 
 * @param team              [in] A team handle.
 *
 * @return The number of PEs in the specified team. 
 *         If the team handle is NVSHMEM_TEAM_INVALID, returns -1.
 */
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