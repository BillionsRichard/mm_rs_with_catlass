/*
    WARNING： 
    
    Our barrier implementation ensures that:
        All operations of all ranks of a team ON EXECUTING/INTERNAL STREAMs before the barrier are visiable to all ranks of the team after the barrier.
        
    Refer to shmem_device_sync.h for using restrictions.
*/

#ifndef SHMEM_HOST_SYNC_H
#define SHMEM_HOST_SYNC_H

#include "acl/acl.h"
#include "shmem_host_def.h"

/**
 * @brief barrier of team on specific stream
 *
 * @param tid              [in] team to do barrier
 * @param stream           [in] stream the barrier will be executed on
 * @return void
 */
void shmem_barrier_on_stream(shmem_team_t tid, aclrtStream stream);

/**
 * @brief barrier of all ranks on specific stream
 *
 * @param stream           [in] stream the barrier will be executed on
 * @return void
 */
void shmem_barrier_all_on_stream(aclrtStream stream);

/**
 * @brief barrier of team on default stream
 *
 * @param tid              [in] team to do barrier
 * @return void
 */
void shmem_barrier(shmem_team_t tid);

/**
 * @brief barrier of all teams on default stream
 *
 * @return void
 */
void shmem_barrier_all();

#endif