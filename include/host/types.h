#ifndef SHMEM_TYPES_H
#define SHMEM_TYPES_H

#include <vector>
#include "stdint.h"
#include "limits.h"
#include "team.h"
#include "smem.h"
#include "smem_shm.h"

#define STATE_SCALAR_INVALID -1
#define SHM_MAX_RANKS 2000
#define SHM_MAX_TEAMS 32

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
    int localMemSize;
    smem_shm_data_op_type dataOpType;
    int timeout;
    int extraSize;
    int globalSize;
} ShmemInitAttr;
typedef ShmemInitAttr ShmemInitAttrT;

// state
typedef struct {
    int version;
    int mype;
    int npes;
    void *heapBase;
    void *p2pHeapBase[SHM_MAX_RANKS];
    size_t heapSize;

    ShmemTeam *teamPools[SHM_MAX_TEAMS];
    long *psyncPool;
    long *syncCounter;

    bool shemeIsShmemInitialized;
    bool shemeIsShmemCreated;
} ShmemDeviceHostState;
typedef ShmemDeviceHostState ShmemDeviceHostStateT;
extern ShmemDeviceHostStateT shmemDeviceHostState;
#endif /*SHMEM_TYPES_H*/