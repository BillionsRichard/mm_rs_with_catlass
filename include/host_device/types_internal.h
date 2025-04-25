#ifndef _TYPES_INTERNAL_
#define _TYPES_INTERNAL_

#include "macros.h"
#include "team_internal.h"

#include "smem.h"
#include "smem_shm.h"

// synchronization
typedef int32_t SyncBit[SYNCBIT_SIZE / sizeof(int32_t)];

// level 0: sync between pipes, use native primitives

// level 1: sync between cores
typedef SyncBit SyncArrayL1[SHM_MAX_CORES_PER_RANK];
typedef SyncBit SyncCounterL1;

// level 2: sync between devices
typedef SyncBit SyncArrayL2[SHM_MAX_RANKS];
typedef SyncBit SyncCounterL2;

// team support
typedef SyncArrayL2 SAPoolL2[SHM_MAX_TEAMS];
typedef SyncCounterL2 SCPoolL2[SHM_MAX_TEAMS];

// level 3: sync between hosts?

#define SHM_MAX_RANKS 2000
#define SHM_MAX_TEAMS 32

#define DEFAULT_FLAG 0
#define STATE_SCALAR_INVALID -1
#define DEFAULT_ID 0

#define DEFAULT_EXTRA_SIZE SHMEM_EXTRA_SIZE

#define SHMEM_DEVICE_HOST_STATE_INITALIZER                                            \
    {                                                                                 \
        (1 << 16) + sizeof(ShmemDeviceHostStateT),  /* version */                     \
            STATE_SCALAR_INVALID,                    /* mype */                       \
            STATE_SCALAR_INVALID,                    /* npes */                       \
            NULL,                                    /* heapBase */                   \
            {NULL},                                  /* p2pHeapBase */                \
            {NULL},                                  /* sdmaHeapBase */               \
            {NULL},                                  /* roceHeapBase */               \
            SIZE_MAX,                                /* heapSize */                   \
            {NULL},                                  /* teamPools */                  \
            NULL,                                    /* psyncPool */                  \
            NULL,                                    /* syncCounter */                \
            false,                                   /* shmem_is_shmem_initialized */ \
            false,                                   /* shmem_is_shmem_created */     \
            {0, 16 * 1024, 0},                       /* shmem_mte_config */           \
    }


// MTEConfig
typedef struct {
    int64_t shmemUB;        // __ubuf__ Ptr, Shmem memcpy needed.
    uint32_t ubSize;        // UB's Size, in Bytes.
    uint32_t eventID;       // TEventID, for Shmem memcpy sync.
} ShmemMTEConfig;

// state
typedef struct {
    int version;
    int mype;
    int npes;
    void *heapBase;
    void *p2pHeapBase[SHM_MAX_RANKS];
    void *sdmaHeapBase[SHM_MAX_RANKS];
    void *roceHeapBase[SHM_MAX_RANKS];
    size_t heapSize;

    ShmemTeam *teamPools[SHM_MAX_TEAMS];
    
    // Using SyncBit instead of basic types to store flag, avoiding concurrent write due to cacheline sharing.
    // Refer to shmemi_barrier.h for more details.
    SyncBit *sPoolL2;
    SyncBit *cPoolL2;

    bool shemeIsShmemInitialized;
    bool shemeIsShmemCreated;

    ShmemMTEConfig mteConfig;
} ShmemDeviceHostState;
typedef ShmemDeviceHostState ShmemDeviceHostStateT;
extern ShmemDeviceHostStateT shmemDeviceHostState;

#endif