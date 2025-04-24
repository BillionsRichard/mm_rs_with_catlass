#ifndef _TYPES_INTERNAL_
#define _TYPES_INTERNAL_

#include "smem_shm.h"
#include "team_internal.h"

#define SHM_MAX_RANKS 2000
#define SHM_MAX_TEAMS 32

#define SHM_MAX_RANKS 2000
#define SHM_MAX_TEAMS 32

#define DEFAULT_FLAG 0
#define STATE_SCALAR_INVALID -1
#define DEFAULT_ID 0
#define DEFAULT_EXTRA_SIZE 0

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
    long *psyncPool;
    long *syncCounter;

    bool shemeIsShmemInitialized;
    bool shemeIsShmemCreated;

    ShmemMTEConfig mteConfig;
} ShmemDeviceHostState;
typedef ShmemDeviceHostState ShmemDeviceHostStateT;
extern ShmemDeviceHostStateT shmemDeviceHostState;

#endif