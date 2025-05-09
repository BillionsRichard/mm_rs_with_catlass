/*
    WARNING： 
    
    1. Barriers can be used only in MIX kernels. The compiler will optimize the kernel to VEC or CUBE if it lacks effective cube instructions (eg. Mmad) or vector instructions (eg: DataCopy). 
    Need compiler updates to remove this feature, or insert Mmad/DataCopy calls manully.
    2. Scalar unit of cube core is not affected by barrier. Make sure don't use that.
*/

#ifndef SHMEM_HOST_SYNC_H
#define SHMEM_HOST_SYNC_H

#include "acl/acl.h"
#include "shmem_host_def.h"

SHMEM_HOST_API void shmem_barrier_on_stream(shmem_team_t tid, aclrtStream stream);

SHMEM_HOST_API void shmem_barrier_all_on_stream(aclrtStream stream);

SHMEM_HOST_API void shmem_barrier(shmem_team_t tid);

SHMEM_HOST_API void shmem_barrier_all();

#endif