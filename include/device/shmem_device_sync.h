/*
    WARNING：Restrictions of Barrier APIs.
    
    1. Barrier APIs can be used only in MIX kernels. The compiler will optimize the kernel to VEC or CUBE if it lacks effective cube instructions (eg. Mmad) or vector instructions (eg: DataCopy). 
    Need compiler updates to remove this feature, or insert Mmad/DataCopy calls manully.

    2. Barrier APIs conflict with SyncAll. Avoid mixing them together.
    
    3. We provide 2 kinds of barrier:
        a. shmem_barrier_xxx
            All operations of all ranks of a team on excuting stream before the barrier are visiable to all ranks of the team after the barrier.
        b. shmemx_barrier_xxx_vec
            All operations of ALL VEC CORES of all ranks of a team on excuting stream before the barrier are visiable to ALL VEC CORES of all ranks of the team after the barrier.
        
        This subtle difference is beneficial to compute-communiction overlapping (usually UNI_DIRECTIONAL dependency), and could achieve better performance. Refer to examples/matmul_allreduce for details.

    4. The scalar unit of cube core is not affected by shmem_barrier_xxx. Make sure don't use that.
*/

#ifndef SHMEM_DEVICE_SYNC_H
#define SHMEM_DEVICE_SYNC_H

#include "host_device/shmem_types.h"
#include "internal/device/sync/shmemi_device_quiet.h"
#include "internal/device/sync/shmemi_device_p2p.h"
#include "internal/device/sync/shmemi_device_barrier.h"

/**
 * @brief barrier of a specific team
 *
 * @param tid              [in] team to do barrier
 * @return void
 */
SHMEM_DEVICE void shmem_barrier(shmem_team_t tid) {
    ShmemiBarrier<false>(tid);
}

/**
 * @brief barrier of all ranks
 *
 * @return void
 */
SHMEM_DEVICE void shmem_barrier_all() {
    shmem_barrier(SHMEM_TEAM_WORLD);
}

/**
 * @brief barrier of a specific team. Different from shmem_barrier that only vector cores participate. Useful in communication-over-compute operators. Cube core may call the api but takes no effect.
 *
 * @param tid              [in] team to do barrier
 * @return void
 */
SHMEM_DEVICE void shmemx_barrier_vec(shmem_team_t tid) {
    ShmemiBarrier<true>(tid);
}

/**
 * @brief barrier of all ranks. Different from shmem_barrier_all that only vector cores participate. Useful in communication-over-compute operators. Cube core may call the api but takes no effect.
 *
 * @param tid              [in] team to do barrier
 * @return void
 */
SHMEM_DEVICE void shmemx_barrier_all_vec() {
    shmemx_barrier_vec(SHMEM_TEAM_WORLD);
}

/**
 * @brief Sync primitive that ensures completion:
            All operations of the calling thread before the primitive are completed.
 *
 * @return void
 */
SHMEM_DEVICE void shmem_quiet() {
    ShmemiQuiet();
}

/**
 * @brief Sync primitive that preservers order: 
            All operations of the calling thread before the primitive are visible to the calling thread after the primitive.
        Implemented same as shmem_quiet().
 *
 * @return void
 */
SHMEM_DEVICE void shmem_fence() {
    ShmemiQuiet();
}

#endif