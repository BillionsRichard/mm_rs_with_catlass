#ifndef SHMEMI_TEAM_H
#define SHMEMI_TEAM_H

#include "macros.h"
#include "types.h"
#include "team.h"

#include "internal/device/shmemi_device_common.h"

SHMEM_AICORE_INLINE 
__gm__ SyncBit *ShmemiGetTeamSyncArrayL2(ShmemTeam *team) {
    uint64_t addr = (uint64_t) getState()->sPoolL2;
    addr += team->teamIdx * SYNC_ARRAY_SIZE_L2;
    return (__gm__ SyncBit *) addr;
}

SHMEM_AICORE_INLINE 
__gm__ SyncBit *ShmemiGetTeamSyncCounterL2(ShmemTeam *team) {
    uint64_t addr = (uint64_t) getState()->cPoolL2;
    addr += team->teamIdx * SYNC_COUNTER_SIZE_L2;
    return (__gm__ SyncBit *) addr;
}

#endif 