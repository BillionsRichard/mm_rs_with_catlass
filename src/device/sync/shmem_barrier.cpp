#include "acl/acl.h"

#include "team.h"
#include "internal/device/sync/shmemi_barrier.h"
#include "internal/device/sync/shmemi_p2p.h"
#include "internal/device/sync/shmemi_quiet.h"

// kernels
SHMEM_AICORE_KERNEL void KShmemBarrier(ShmemTeam_t tid) {
    ShmemiBarrier(tid);
} 

// interfaces
void ShmemBarrier(ShmemTeam_t tid) {
    // TODO: clear all working streams

    // using default stream to do barrier
    KShmemBarrier<<<1, nullptr, nullptr>>>(tid);
    aclrtSynchronizeStream(nullptr);
}

void ShmemBarrierAll() {
    ShmemBarrier(SHMEM_TEAM_WORLD);
}