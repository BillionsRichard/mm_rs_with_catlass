#ifndef _SHMEM_DEVICE_API_
#define _SHMEM_DEVICE_API_

#define SHM_MAX_RANKS 2000
#define SHM_MAX_TEAMS 32

typedef struct {
    int mype;           // team view, [0, size]
    int start;          // global view, [0, npes]
    int stride;         // global view, [1, npes - 1]
    int size;           // team view
    int teamIdx;
} ShmemTeam;

enum {
    SHMEM_TEAM_INVALID = -1,
    SHMEM_TEAM_WORLD = 0,
    SHMEM_TEAM_WORLD_INDEX = 0
};

typedef int ShmemTeam_t;

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

#include "scalar_p.hpp"
#include "team.hpp"

#endif