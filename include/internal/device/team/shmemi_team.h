#ifndef SHMEMI_TEAM_H
#define SHMEMI_TEAM_H

#include "macros.h"
#include "types.h"
#include "team.h"

#include "internal/device/shmemi_device_common.h"

SHMEM_AICORE_INLINE __gm__ SyncBit *ShmemiGetTeamSyncArray(ShmemTeam *team) {
    uint64_t addr = reinterpret_cast<uint64_t>(getState()->syncArray);
    addr += team->teamIdx * SYNC_ARRAY_SIZE_PER_TEAM;
    return reinterpret_cast<__gm__ SyncBit *>(addr);
}

SHMEM_AICORE_INLINE __gm__ SyncBit *ShmemiGetTeamSyncCounter(ShmemTeam *team) {
    uint64_t addr = reinterpret_cast<uint64_t>(getState()->syncCounter);
    addr += team->teamIdx * SYNCBIT_SIZE;
    return reinterpret_cast<__gm__ SyncBit *>(addr);
}

#endif 