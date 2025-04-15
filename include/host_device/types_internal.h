#ifndef _TYPES_INTERNAL_
#define _TYPES_INTERNAL_

#include "team_internal.h"

#define SHM_MAX_RANKS 2000
#define SHM_MAX_TEAMS 32

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

#endif