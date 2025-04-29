#ifndef SHMEMI_TYPES_H
#define SHMEMI_TYPES_H

#define SHMEM_MAX_CORES_PER_RANK 32
#define SHMEM_MAX_RANKS 2048
#define SHMEM_MAX_TEAMS 32

/* arch related */
#define SCALAR_DATA_CACHELINE_SIZE 64
#define L2_CACHELINE_SIZE 512

#define SHMEM_PAGE_SIZE (1024UL * 1024UL * 2)

#define ALIGH_TO(size, page) ((size + page - 1) / page * page)

/* synchonization related */ 
#define SHMEMI_SYNCBIT_SIZE SCALAR_DATA_CACHELINE_SIZE

// npu level sync
#define SYNC_ARRAY_SIZE (SHMEMI_SYNCBIT_SIZE * SHMEM_MAX_RANKS)
#define SYNC_COUNTER_SIZE SHMEMI_SYNCBIT_SIZE
#define SYNC_POOL_SIZE (SYNC_ARRAY_SIZE * SHMEM_MAX_TEAMS)
#define SYNC_COUNTERS_SIZE (SYNC_COUNTER_SIZE * SHMEM_MAX_TEAMS)

// Total extra
#define SHMEM_EXTRA_SIZE_UNALIGHED SYNC_POOL_SIZE
#define SHMEM_EXTRA_SIZE ALIGH_TO(SHMEM_EXTRA_SIZE_UNALIGHED, SHMEM_PAGE_SIZE)

// synchronization
typedef int32_t ShmemiSyncBit[SHMEMI_SYNCBIT_SIZE / sizeof(int32_t)];

// Team
typedef struct {
    int mype;           // team view, [0, size]
    int start;          // global view, [0, npes]
    int stride;         // global view, [1, npes - 1]
    int size;           // team view
    int teamIdx;
} ShmemiTeam;

// MTEConfig
typedef struct {
    int64_t shmemUB;        // __ubuf__ Ptr, Shmem memcpy needed.
    uint32_t ubSize;        // UB's Size, in Bytes.
    uint32_t eventID;       // TEventID, for Shmem memcpy sync.
} ShmemiMTEConfig;

// state
typedef struct {
    int version;
    int mype;
    int npes;
    void *heapBase;
    void *p2pHeapBase[SHMEM_MAX_RANKS];
    void *sdmaHeapBase[SHMEM_MAX_RANKS];
    void *roceHeapBase[SHMEM_MAX_RANKS];
    size_t heapSize;

    ShmemiTeam *teamPools[SHMEM_MAX_TEAMS];
    
    // Using ShmemiSyncBit instead of basic types to ShmemiStore flag, avoiding concurrent write due to cacheline sharing.
    // Refer to shmemi_barrier.h for more details.
    ShmemiSyncBit *syncPool;
    ShmemiSyncBit *syncCounter;

    bool shemeIsShmemInitialized;
    bool shemeIsShmemCreated;

    ShmemiMTEConfig mteConfig;
} ShmemiDeviceHostState;

#endif