#ifndef SHMEM_SYNC_H
#define SHMEM_SYNC_H

#include "macros.h"
#include "team.h"
#include "internal/device/sync/shmemi_quiet.h"
#include "internal/device/sync/shmemi_p2p.h"
#include "internal/device/sync/shmemi_barrier.h"

SHMEM_AICORE_INLINE void ShmemBarrier(ShmemTeam_t tid) {
    ShmemiBarrier(tid);
}

SHMEM_AICORE_INLINE void ShmemBarrierAll() {
    ShmemBarrier(SHMEM_TEAM_WORLD);
}

SHMEM_AICORE_INLINE void ShmemQuiet() {
    ShmemiQuiet();
}

SHMEM_AICORE_INLINE void ShmemFence() {
    ShmemiQuiet();
}

#endif