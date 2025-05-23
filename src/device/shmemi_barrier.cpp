#include "acl/acl.h"
#include "shmemi_device_intf.h"
#include "device/shmem_device_sync.h"

// kernels
SHMEM_GLOBAL void KShmemBarrier(int32_t tid) {
    shmemi_barrier<false>(tid);
} 

// interfaces
int32_t ShmemiBarrierOnStream(shmem_team_t tid, aclrtStream stream) {
    // TODO: clear all internal working streams

    // call barrier kernel
    KShmemBarrier<<<1, nullptr, stream>>>((int32_t) tid);
    return aclrtSynchronizeStream(stream);
}