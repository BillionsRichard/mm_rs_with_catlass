#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>

using namespace std;

#include "acl/acl.h"
#include "shmemi_host_intf.h"

ShmemiTeam shmemTeamWorld;
ShmemiTeam *shmemiDeviceTeamWorld;
ShmemiTeam **shmemTeamPool;

long *shmemPsyncPool;
long *shmemSyncCounter;
long *poolAvail;

void DeviceTeamUpdate(int teamIdx, ShmemiTeam *hostTeamPtr)
{
    // devicePtr Malloc
    void* teamPtr = NULL;
    aclrtMalloc(&teamPtr, sizeof(ShmemiTeam), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy((ShmemiTeam *)teamPtr, sizeof(ShmemiTeam), hostTeamPtr, sizeof(ShmemiTeam), ACL_MEMCPY_HOST_TO_DEVICE);
    gState.teamPools[teamIdx] = (ShmemiTeam *)teamPtr;
}

void DeviceTeamDestroy(int teamIdx)
{
    // devicePtr Free
    ShmemiTeam *deviceTeamPtr = gState.teamPools[teamIdx];
    aclrtFree((void *)deviceTeamPtr);
    gState.teamPools[teamIdx] = nullptr;
}

int ShmemiTeamInit(int rank, int size)
{
    /* Initialize SHMEM_TEAM_WORLD */
    shmemTeamWorld.teamIdx = 0;
    shmemTeamWorld.start = 0;
    shmemTeamWorld.stride = 1;
    shmemTeamWorld.size = size;       // TODO state->npes
    shmemTeamWorld.mype = rank;       // TODO state->mype

    int shmemMaxTeams = SHMEM_MAX_TEAMS;
    shmemTeamPool = (ShmemiTeam **)calloc(shmemMaxTeams, sizeof(ShmemiTeam *));
    if (shmemTeamPool == nullptr) {
        std::cout << "shmemTeamPool calloc failed!" << std::endl;
        return 1;
    }
    for (int i = 0; i < shmemMaxTeams; i++) {
        shmemTeamPool[i] = nullptr;
    }
    shmemTeamPool[shmemTeamWorld.teamIdx] = &shmemTeamWorld;
    DeviceTeamUpdate(shmemTeamWorld.teamIdx, &shmemTeamWorld);

    poolAvail = (long *)calloc(shmemMaxTeams, sizeof(long));
    poolAvail[0] = 1;

    /* Initialize TEAM SYNC */    /* Initialize TEAM SYNC */
    gState.syncPool = (ShmemiSyncBit *)ShmemMalloc(SYNC_POOL_SIZE);
    aclrtMemset((void *) gState.syncPool, SYNC_POOL_SIZE, 0, SYNC_POOL_SIZE);

    aclrtMalloc((void **)&(gState.syncCounter), SYNC_COUNTERS_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    ShmemiMemset((int32_t *) gState.syncCounter, SYNC_COUNTERS_SIZE / sizeof(int32_t), 1);

    return 0;
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
        ShmemTeam parentTeam,
        int peStart, int peStride, int peSize,
        ShmemTeam &newTeam)
{
    newTeam = SHMEM_TEAM_INVALID;

    ShmemiTeam *myteam = nullptr;
    myteam = (ShmemiTeam *)calloc(1, sizeof(ShmemiTeam));

    ShmemiTeam *srcTeam = shmemTeamPool[parentTeam];

    int globalPE = srcTeam->mype;
    int globalPeStart = srcTeam->start + peStart * srcTeam->stride;
    int globalPeStride = srcTeam->stride * peStride;
    int globalPeEnd = globalPeStart + globalPeStride * (peSize - 1);

    if (peStart < 0 || peStart >= srcTeam->size || peSize <= 0 || peSize > srcTeam->size || peStride < 1) {
        std::cout << "InValid team create !" << std::endl;
        return -1;
    }

    if (globalPeStart >= shmemTeamPool[0]->size || globalPeEnd >= shmemTeamPool[0]->size) {
        std::cout << "InValid team create !" << std::endl;
        return -1;
    }

    myteam->mype = (globalPE - globalPeStart) / globalPeStride;

    if (globalPE < globalPeStart || (globalPE - globalPeStart)  % globalPeStride || myteam->mype >= peSize) {
        std::cout << "InValid team create !" << std::endl;
        return -1;
    }

    myteam->start = globalPeStart;
    myteam->stride = globalPeStride;
    myteam->size = peSize;

    myteam->teamIdx = FirstFreeIdxFetch();
    if (myteam->teamIdx == -1) {
        std::cout << "EXCEED MAX_TEAM SIZE !!" << std::endl;
        return -1;
    }
    shmemTeamPool[myteam->teamIdx] = myteam;
    DeviceTeamUpdate(myteam->teamIdx, myteam);
    UpdateDeviceState();

    newTeam = myteam->teamIdx;
    return 1;
}


int ShmemTeamTranslatePE(
    ShmemTeam srcTeam, int srcPe,
    ShmemTeam destTeam)
{
    if (srcTeam == SHMEM_TEAM_INVALID || destTeam == SHMEM_TEAM_INVALID) return -1;
    ShmemiTeam *srcTeamPtr = shmemTeamPool[srcTeam];
    ShmemiTeam *destTeamPtr = shmemTeamPool[destTeam];

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


void ShmemTeamDestroy(ShmemTeam team)
{
    if (team == -1) {
        return;
    }
    poolAvail[team] = 0;
    shmemTeamPool[team] = nullptr;
    DeviceTeamDestroy(team);

    return;
}


int ShmemiTeamFinalize() {
    /* Destroy all undestroyed teams*/
    int shmemMaxTeams = SHMEM_MAX_TEAMS;
    for (int i = 0; i < shmemMaxTeams; i++) {
        if (shmemTeamPool[i] != NULL) ShmemTeamDestroy((ShmemTeam)i);
    }

    free(shmemTeamPool);
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


int ShmemTeamMype(ShmemTeam team)
{
    if (team == SHMEM_TEAM_INVALID)
        return -1;
    else
        return shmemTeamPool[team]->mype;
}


int ShmemTeamNpes(ShmemTeam team)
{
    if (team == SHMEM_TEAM_INVALID)
        return -1;
    else
        return shmemTeamPool[team]->size;
}