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
            false,                                   /* sheme_is_shmem_initialized */ \
            false,                                   /* sheme_is_shmem_created */     \
    }

// attr
typedef struct {
    int version;
    int id;
    const char* ipPort;
    int myRank;
    int nRanks;
    int deviceId;
    uint64_t localMemSize;
    smem_shm_data_op_type dataOpType;
    int timeout;
    uint64_t extraSize;
    uint64_t globalSize;
} ShmemInitAttr;
typedef ShmemInitAttr ShmemInitAttrT;

typedef ShmemDeviceHostState ShmemDeviceHostStateT;
extern ShmemDeviceHostStateT shmemDeviceHostState;
#endif /*SHMEM_TYPES_H*/