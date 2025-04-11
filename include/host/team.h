#ifndef SHMEM_TEAM_H
#define SHMEM_TEAM_H

#include <climits>
#include <cstdlib>
#include <cstdbool>
#include <acl/acl.h>

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

int ShmemTeamInit(int rank, int size);                    // TODO, No inputs

int ShmemTeamFinalize();

int ShmemTeamSplitStrided(ShmemTeam_t parentTeam, int peStart, int peStride, int peSize, ShmemTeam_t &newTeam);

int ShmemTeamTranslatePE(ShmemTeam_t srcTeam, int srcPe, ShmemTeam_t destTeam);

void ShmemTeamDestroy(ShmemTeam_t team);

int ShmemMype();

int ShmemNpes();

int ShmemTeamMype(ShmemTeam_t team);

int ShmemTeamNpes(ShmemTeam_t team);

#endif