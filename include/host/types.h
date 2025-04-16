#ifndef SHMEM_TYPES_H
#define SHMEM_TYPES_H

#include <vector>
#include "stdint.h"
#include "limits.h"
#include "team.h"
#include "smem.h"
#include "smem_shm.h"

#include "shmem_internal.h"

// attr
typedef struct {
    int myRank;
    int nRanks;
    uint64_t localMemSize;
} ShmemInitAttr;
typedef ShmemInitAttr ShmemInitAttrT;

#endif /*SHMEM_TYPES_H*/