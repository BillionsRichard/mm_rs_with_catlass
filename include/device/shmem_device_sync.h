/*
    WARNING： 
    
    1. Barriers can be used only in MIX kernels. The compiler will optimize the kernel to VEC or CUBE if it lacks effective cube instructions (eg. Mmad) or vector instructions (eg: DataCopy). 
    Need compiler updates to remove this feature, or insert Mmad/DataCopy calls manully.
    2. Scalar unit of cube core is not affected by barrier. Make sure don't use that.
*/

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