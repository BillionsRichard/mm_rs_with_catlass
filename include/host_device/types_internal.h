#ifndef _TYPES_INTERNAL_
#define _TYPES_INTERNAL_

#include "smem_shm.h"
#include "team_internal.h"

#define SHM_MAX_RANKS 2000
#define SHM_MAX_TEAMS 32

#define DEFAULT_FLAG 0
#define STATE_SCALAR_INVALID -1
#define DEFAULT_ID 0
#define DEFAULT_IP_PORT "tcp://127.0.0.1:8666"
#define DEFAULT_TIMEOUT 30
#define DEFAULT_EXTRA_SIZE 0

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

#define SHMEM_COMM_ATTR                                                               \
    {                                                                                 \
        (1 << 16) + sizeof(ShmemDeviceHostStateT),  /* version */                     \
            DEFAULT_ID,                             /* id */                          \
            DEFAULT_IP_PORT,                        /* ipPort */                      \
            STATE_SCALAR_INVALID,                   /* deviceId */                    \
            SMEMS_DATA_OP_MTE,                      /* dataOpType */                  \
            DEFAULT_TIMEOUT,                        /* timeout */                     \
            DEFAULT_EXTRA_SIZE,                     /* extraSize */                   \
            DEFAULT_EXTRA_SIZE,                     /* globalSize */                  \
            DEFAULT_FLAG                            /* flag */                        \
    }

// MTEConfig
typedef struct {
    int64_t tmpUb;          // __ubuf__ Ptr
    uint32_t ubSize;        // Bytes
    uint32_t eventID;       // TEventID, for device sync
} ShmemMTEConfig;

// commattr
typedef struct {
    int version;
    int id;
    const char* ipPort;
    int32_t deviceId;
    smem_shm_data_op_type dataOpType;
    int timeout;
    uint64_t extraSize;
    uint64_t globalSize;
    uint32_t flag;
} ShmemCommAttr;
typedef ShmemCommAttr ShmemCommAttrT;
extern ShmemCommAttrT shmemCommAttr;

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

    ShmemMTEConfig mteConfig;
} ShmemDeviceHostState;
typedef ShmemDeviceHostState ShmemDeviceHostStateT;
extern ShmemDeviceHostStateT shmemDeviceHostState;

#endif