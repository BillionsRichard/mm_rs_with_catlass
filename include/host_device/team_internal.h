#ifndef _TEAM_INTERNAL_
#define _TEAM_INTERNAL_

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

#endif