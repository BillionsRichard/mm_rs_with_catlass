/*
    WARNING： 
    
    1. Barriers can be used only in MIX kernels. The compiler will optimize the kernel to VEC or CUBE if it lacks effective cube instructions (eg. Mmad) or vector instructions (eg: DataCopy). 
    Need compiler updates to remove this feature, or insert Mmad/DataCopy calls manully.
    
    2. Unlike semantic of legacy barrier:
            All operations of all ranks of a team before the barrier are visiable to all ranks of the team after the barrier.
        Our implementation ensures that:
            All operations of ALL VEC CORES of all ranks of a team ON EXCUTING STREAM before the barrier are visiable to ALL VEC CORES of all ranks of the team after the barrier.
        
        This subtle difference is beneficial to compute-communiction overlapping (usually UNI_DIRECTIONAL dependency), and could achieve better performance.
        
        Example of Matmul_AllReduce kernel:
            SHMEM_DEVICE void matmul_allreduce() {
                if ASCEND_IS_AIV 
                    CrossCoreSetFlag<0x02, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);

                for (xxx) {
                    if ASCEND_IS_AIC {
                        CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
                        matmul();
                        CrossCoreSetFlag<0x02, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
                    }

                    if ASCEND_IS_AIV {
                        CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);

                        shmem_barrier_xxx();
                        reduce_scatter();
                        shmem_barrier_xxx();
                        all_gather();

                        CrossCoreSetFlag<0x02, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
                    }
                }
                
                if ASCEND_IS_AIC
                    CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
            }
        Moreover, double buffer can be used to increase parallism.

    3. In case that legacy barrier is needed, it can be implemented as below:
            SHMEM_DEVICE void legacy_barrier() {
                if ASCEND_IS_AIC {
                    PipeBarrier<PIPE_ALL>();
                    CrossCoreSetFlag<0x02, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
                    CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
                }

                if ASCEND_IS_AIV {
                    CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
                    shmem_barrier_xxx();
                    CrossCoreSetFlag<0x02, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
                }
            }
        Even though, scalar unit of cube core is not affected by barrier. Make sure don't use that.

    4. Barrier APIs conflict with SyncAll. Avoid mixing them together.
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
    ShmemiBarrier(tid);
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