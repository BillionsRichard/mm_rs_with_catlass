#ifndef SHMEM_TYPES_H
#define SHMEM_TYPES_H

#include <vector>
#include "stdint.h"
#include "limits.h"
#include "team.h"
#include "smem.h"
#include "smem_shm.h"

#include "shmem_internal.h"

#define STATE_SCALAR_INVALID -1

#define SHMEM_TEAM_INITALIZER                                                         \
    {                                                                                 \
        (1 << 16) + sizeof(ShmemDeviceHostStateT), /* version */                      \
    }

#define SHMEM_DEVICE_HOST_STATE_INITALIZER                                            \
    {                                                                                 \
        (1 << 16) + sizeof(ShmemDeviceHostStateT),  /* version */                     \
            STATE_SCALAR_INVALID,                    /* mype */                       \
            STATE_SCALAR_INVALID,                    /* npes */                       \
            NULL,                                    /* heapBase */                   \
            {NULL},                                  /* p2pHeapBase */                \
            SIZE_MAX,                                /* heapSize */                   \
            {NULL},                                  /* teamPools */                  \
            NULL,                                    /* psyncPool */                  \
            NULL,                                    /* syncCounter */                \
            false,                                   /* shmem_is_shmem_initialized */ \
            false,                                   /* shmem_is_shmem_created */     \
            {0, 0, 0},                               /* shmem_mte_config */           \
    }

// attr
typedef struct {
    int myRank;
    int nRanks;
    uint64_t localMemSize;
} ShmemInitAttr;
typedef ShmemInitAttr ShmemInitAttrT;

#endif /*SHMEM_TYPES_H*/