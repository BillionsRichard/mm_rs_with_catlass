#include "acl/acl.h"

#include "shmemi_host_intf.h"
#include "shmem_device_api.h"

// kernels
SHMEM_GLOBAL void KShmemBarrier(int tid) {
    ShmemiBarrier(tid);
} 

// interfaces
void ShmemBarrierOnStream(ShmemTeam tid, aclrtStream stream) {
    // TODO: clear all internal working streams

    // call barrier kernel
    KShmemBarrier<<<1, nullptr, stream>>>((int) tid);
    CHECK_ACL(aclrtSynchronizeStream(stream));
}

void ShmemBarrierAllOnStream(aclrtStream stream) {
    ShmemBarrierOnStream(SHMEM_TEAM_WORLD, stream);
}

void ShmemBarrier(ShmemTeam tid) {
    // using default stream to do barrier
    ShmemBarrierOnStream(tid, nullptr);
}

void ShmemBarrierAll() {
    ShmemBarrier(SHMEM_TEAM_WORLD);
}