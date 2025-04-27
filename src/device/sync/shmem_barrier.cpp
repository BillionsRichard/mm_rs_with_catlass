#include "acl/acl.h"

#include "macros.h"
#include "data_utils.h"
#include "team.h"
#include "internal/device/sync/shmemi_barrier.h"
#include "internal/device/sync/shmemi_p2p.h"
#include "internal/device/sync/shmemi_quiet.h"

// kernels
SHMEM_AICORE_KERNEL void KShmemBarrier(int tid) {
    CVGuard();
    ShmemiBarrier(tid);
} 

// interfaces
void ShmemBarrierOnStream(ShmemTeam_t tid, aclrtStream stream) {
    // TODO: clear all internal working streams

    // call barrier kernel
    KShmemBarrier<<<1, nullptr, stream>>>((int) tid);
    CHECK_ACL(aclrtSynchronizeStream(stream));
}

void ShmemBarrierAllOnStream(aclrtStream stream) {
    ShmemBarrierOnStream(SHMEM_TEAM_WORLD, stream);
}

void ShmemBarrier(ShmemTeam_t tid) {
    // using default stream to do barrier
    ShmemBarrierOnStream(tid, nullptr);
}

void ShmemBarrierAll() {
    ShmemBarrier(SHMEM_TEAM_WORLD);
}