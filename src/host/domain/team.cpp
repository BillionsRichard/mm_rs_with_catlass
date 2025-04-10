#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace std;

#include "team.h"

#define SHMEM_MAX_TEAMS 32

ShmemTeam shmemTeamWorld;
ShmemTeam *shmemiDeviceTeamWorld;
ShmemTeam **shmemTeamPool;

long *shmemPsyncPool;
long *shmemSyncCounter;
long *poolAvail;

int ShmemTeamInit(int rank, int size)
{
    /* Initialize SHMEM_TEAM_WORLD */
    shmemTeamWorld.teamIdx = 0;
    shmemTeamWorld.start = 0;
    shmemTeamWorld.stride = 1;
    shmemTeamWorld.size = size;       // TODO state->npes
    shmemTeamWorld.mype = rank;       // TODO state->mype

    int shmemMaxTeams = SHMEM_MAX_TEAMS;
    shmemTeamPool = (ShmemTeam **)calloc(shmemMaxTeams, sizeof(ShmemTeam *));
    for (int i = 0; i < shmemMaxTeams; i++) {
        shmemTeamPool[i] = nullptr;
    }
    shmemTeamPool[shmemTeamWorld.teamIdx] = &shmemTeamWorld;

    poolAvail = (long *)calloc(shmemMaxTeams, sizeof(long));
    poolAvail[0] = 1;

    /* Initialize TEAM SYNC */
    long psyncLen = shmemMaxTeams * 1024;
    // shmemPsyncPool = (long *)ShmemMalloc(sizeof(long) * psyncLen);
    // shmemSyncCounter = (long *)ShmemMalloc(2 * shmemMaxTeams * sizeof(long));

    return 1;
}


int FirstFreeIdxFetch()
{
    int shmemMaxTeams = SHMEM_MAX_TEAMS;
    for (int i = 0; i < shmemMaxTeams; i++) {
        if (poolAvail[i] == 0) {
            poolAvail[i] = 1;
            return i;
        }
    }
    return -1;
}


int ShmemTeamSplitStrided(
        ShmemTeam_t parentTeam,
        int PE_start, int PE_stride, int PE_size,
        ShmemTeam_t &newTeam)
{
    newTeam = SHMEM_TEAM_INVALID;

    ShmemTeam *myteam = nullptr;
    myteam = (ShmemTeam *)calloc(1, sizeof(ShmemTeam));

    ShmemTeam *srcTeam = shmemTeamPool[parentTeam];

    int global_pe = srcTeam->mype;
    int global_PE_start = srcTeam->start + PE_start * srcTeam->stride;
    int global_PE_stride = srcTeam->stride * PE_stride;
    int global_PE_end = global_PE_start + global_PE_stride * (PE_size - 1);

    if (PE_start < 0 || PE_start >= srcTeam->size || PE_size <= 0 || PE_size > srcTeam->size || PE_stride < 1) {
        // std::cout << "InValid team create !" << std::endl;                  // TODO LOG
        return -1;
    }

    if (global_PE_start >= shmemTeamPool[0]->size || global_PE_end >= shmemTeamPool[0]->size) {
        // std::cout << "InValid team create !" << std::endl;                  // TODO LOG
        return -1;
    }

    myteam->mype = (global_pe - global_PE_start) / global_PE_stride;

    if (global_pe < global_PE_start || (global_pe - global_PE_start)  % global_PE_stride || myteam->mype >= PE_size) {
        // std::cout << "InValid team create !" << std::endl;                  // TODO LOG
        return -1;
    }

    myteam->start = global_PE_start;
    myteam->stride = global_PE_stride;
    myteam->size = PE_size;

    myteam->teamIdx = FirstFreeIdxFetch();
    if (myteam->teamIdx == -1) {
        // std::cout << "EXCEED MAX_TEAM SIZE !!" << std::endl;                  // TODO LOG
        return -1;
    }
    shmemTeamPool[myteam->teamIdx] = myteam;

    newTeam = myteam->teamIdx;
    return 1;
}


int ShmemTeamTranslate_pe(
    ShmemTeam_t srcTeam, int srcPe,
    ShmemTeam_t destTeam)
{
    if (srcTeam == SHMEM_TEAM_INVALID || destTeam == SHMEM_TEAM_INVALID) return -1;
    ShmemTeam *srcTeamPtr = shmemTeamPool[srcTeam];
    ShmemTeam *destTeamPtr = shmemTeamPool[destTeam];

    if (srcPe > srcTeamPtr->size) return -1;

    int global_pe = srcTeamPtr->start + srcPe * srcTeamPtr->stride;
    int PE_start = destTeamPtr->start;
    int PE_stride = destTeamPtr->stride;
    int PE_size = destTeamPtr->size;

    int n = (global_pe - PE_start) / PE_stride;
    if (global_pe < PE_start || (global_pe - PE_start) % PE_stride || n >= PE_size)
        return -1;
    
    return n;
}


void ShmemTeamDestroy(ShmemTeam_t team)
{
    if (team == -1) {
        return;
    }
    poolAvail[team] = 0;
    shmemTeamPool[team] = nullptr;

    return;
}


int ShmemTeamFinalize() {
    /* Destroy all undestroyed teams*/
    int shmemMaxTeams = shmemMaxTeams;
    for (int i = 0; i < shmemMaxTeams; i++) {
        if (shmemTeamPool[i] != NULL) ShmemTeamDestroy((ShmemTeam_t)i);
    }

    free(shmemTeamPool);

    // ShmemFree(shmemPsyncPool);
    // ShmemFree(shmemSyncCounter);
    free(poolAvail);
    return 0;
}


int ShmemMype()
{
    return shmemTeamPool[0]->mype;
}


int ShmemNpes()
{
    return shmemTeamPool[0]->size;
}


int ShmemTeamMype(ShmemTeam_t team)
{
    if (team == SHMEM_TEAM_INVALID)
        return -1;
    else
        return shmemTeamPool[team]->mype;
}


int ShmemTeamNpes(ShmemTeam_t team)
{
    if (team == SHMEM_TEAM_INVALID)
        return -1;
    else
        return shmemTeamPool[team]->size;
}