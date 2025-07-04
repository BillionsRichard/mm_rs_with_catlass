#include "acl/acl.h"
#include "kernel_operator.h"

#include "shmem_api.h"

// kernels
SHMEM_GLOBAL void k_shmem_barrier(int32_t tid) {

    shmemi_barrier<false>(tid);
} 

// interfaces
int32_t shmemi_barrier_on_stream(shmem_team_t tid, aclrtStream stream) {
    // TODO: clear all internal working streams

    // call barrier kernel
    k_shmem_barrier<<<1, nullptr, stream>>>((int32_t) tid);
    return aclrtSynchronizeStream(stream);
}