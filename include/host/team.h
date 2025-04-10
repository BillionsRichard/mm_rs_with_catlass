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
    int team_idx;
} shmem_team;

enum {
    SHMEM_TEAM_INVALID = -1,
    SHMEM_TEAM_WORLD = 0,
    SHMEM_TEAM_WORLD_INDEX = 0
};

typedef int shmem_team_t;

int shmem_team_init(int rank, int size);                    // TODO, No inputs

int shmem_team_finalize();

int shmem_team_split_strided(shmem_team_t parent_team, int PE_start, int PE_stride, int PE_size, shmem_team_t &new_team);

int shmem_team_translate_pe(shmem_team_t src_team, int src_pe, shmem_team_t dest_team);

void shmem_team_destroy(shmem_team_t team);

int shmem_mype();

int shmem_n_pes();

int shmem_team_mype(shmem_team_t team);

int shmem_team_n_pes(shmem_team_t team);

#endif