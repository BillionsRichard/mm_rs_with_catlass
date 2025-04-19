#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>

using namespace std;

#include "init.h"
#include "team.h"
#include "types.h"
#include "acl/acl.h"

#define SHMEM_MAX_TEAMS 32

extern ShmemDeviceHostStateT shmemDeviceHostState;

ShmemTeam shmemTeamWorld;
ShmemTeam *shmemiDeviceTeamWorld;
ShmemTeam **shmemTeamPool;

long *shmemPsyncPool;
long *shmemSyncCounter;
long *poolAvail;

void DeviceTeamUpdate(int teamIdx, ShmemTeam *hostTeamPtr)
{
    // devicePtr Malloc
    void* teamPtr = NULL;
    aclrtMalloc(&teamPtr, sizeof(ShmemTeam), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy((ShmemTeam *)teamPtr, sizeof(ShmemTeam), hostTeamPtr, sizeof(ShmemTeam), ACL_MEMCPY_HOST_TO_DEVICE);
    shmemDeviceHostState.teamPools[teamIdx] = (ShmemTeam *)teamPtr;
}

void DeviceTeamDestroy(int teamIdx)
{
    // devicePtr Free
    ShmemTeam *deviceTeamPtr = shmemDeviceHostState.teamPools[teamIdx];
    aclrtFree((void *)deviceTeamPtr);
    shmemDeviceHostState.teamPools[teamIdx] = nullptr;
}

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

    /* Initialize TEAM SYNC */
    // TODO: use ShmemMalloc; use vector unit to initialize
    shmemDeviceHostState.syncArray = (SyncBit *) shmemDeviceHostState.p2pHeapBase[rank];
    shmemDeviceHostState.syncCounter = (SyncBit *) ((uint64_t)shmemDeviceHostState.p2pHeapBase[rank] + SYNC_ARRAY_SIZE);
    SyncBit set = {1, 0, 0, 0, 0, 0, 0, 0};
    SyncBit clear = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < SYNC_ARRAY_SIZE / SYNCBIT_SIZE; i++) {
        aclrtMemcpy(shmemDeviceHostState.syncArray + i, SYNCBIT_SIZE, &clear, SYNCBIT_SIZE, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    for (int i = 0; i < SYNC_COUNTER_SIZE / SYNCBIT_SIZE; i++) {
        aclrtMemcpy(shmemDeviceHostState.syncCounter + i, SYNCBIT_SIZE, &set, SYNCBIT_SIZE, ACL_MEMCPY_HOST_TO_DEVICE);
    }
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
        ShmemTeam_t parentTeam,
        int peStart, int peStride, int peSize,
        ShmemTeam_t &newTeam)
{
    newTeam = SHMEM_TEAM_INVALID;

    ShmemTeam *myteam = nullptr;
    myteam = (ShmemTeam *)calloc(1, sizeof(ShmemTeam));

    ShmemTeam *srcTeam = shmemTeamPool[parentTeam];

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
    ShmemTeam_t srcTeam, int srcPe,
    ShmemTeam_t destTeam)
{
    if (srcTeam == SHMEM_TEAM_INVALID || destTeam == SHMEM_TEAM_INVALID) return -1;
    ShmemTeam *srcTeamPtr = shmemTeamPool[srcTeam];
    ShmemTeam *destTeamPtr = shmemTeamPool[destTeam];

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


void ShmemTeamDestroy(ShmemTeam_t team)
{
    if (team == -1) {
        return;
    }
    poolAvail[team] = 0;
    shmemTeamPool[team] = nullptr;
    DeviceTeamDestroy(team);

    return;
}


int ShmemTeamFinalize() {
    /* Destroy all undestroyed teams*/
    int shmemMaxTeams = shmemMaxTeams;
    for (int i = 0; i < shmemMaxTeams; i++) {
        if (shmemTeamPool[i] != NULL) ShmemTeamDestroy((ShmemTeam_t)i);
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