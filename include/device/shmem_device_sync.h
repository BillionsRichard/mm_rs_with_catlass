#ifndef SHMEM_DEVICE_SYNC_H
#define SHMEM_DEVICE_SYNC_H

#include "../host_device/shmem_types.h"

#include "internal/device/sync/shmemi_device_quiet.h"
#include "internal/device/sync/shmemi_device_p2p.h"
#include "internal/device/sync/shmemi_device_barrier.h"

SHMEM_DEVICE void ShmemBarrier(ShmemTeam tid) {
    ShmemiBarrier(tid);
}

SHMEM_DEVICE void ShmemBarrierAll() {
    ShmemBarrier(SHMEM_TEAM_WORLD);
}

SHMEM_DEVICE void ShmemQuiet() {
    ShmemiQuiet();
}

SHMEM_DEVICE void ShmemFence() {
    ShmemiQuiet();
}

#endif