#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace std;

#include "team.h"

#define SHMEM_MAX_TEAMS 32

shmem_team shmem_team_world;
shmem_team *shmemi_device_team_world;
shmem_team **shmem_team_pool;

long *shmem_psync_pool;
long *shmem_sync_counter;
long *pool_avail;

int shmem_team_init(int rank, int size)
{
    /* Initialize SHMEM_TEAM_WORLD */
    shmem_team_world.team_idx = 0;
    shmem_team_world.start = 0;
    shmem_team_world.stride = 1;
    shmem_team_world.size = size;       // TODO state->npes
    shmem_team_world.mype = rank;       // TODO state->mype

    int shmem_max_teams = SHMEM_MAX_TEAMS;
    shmem_team_pool = (shmem_team **)calloc(shmem_max_teams, sizeof(shmem_team *));
    for (int i = 0; i < shmem_max_teams; i++) {
        shmem_team_pool[i] = nullptr;
    }
    shmem_team_pool[shmem_team_world.team_idx] = &shmem_team_world;

    pool_avail = (long *)calloc(shmem_max_teams, sizeof(long));
    pool_avail[0] = 1;

    /* Initialize TEAM SYNC */
    long psync_len = shmem_max_teams * 1024;
    // shmem_psync_pool = (long *)shmem_malloc(sizeof(long) * psync_len);
    // shmem_sync_counter = (long *)shmem_malloc(2 * shmem_max_teams * sizeof(long));

    return 1;
}


int first_free_idx_fetch()
{
    int shmem_max_teams = SHMEM_MAX_TEAMS;
    for (int i = 0; i < shmem_max_teams; i++) {
        if (pool_avail[i] == 0) {
            pool_avail[i] = 1;
            return i;
        }
    }
    return -1;
}


int shmem_team_split_strided(
        shmem_team_t parent_team,
        int PE_start, int PE_stride, int PE_size,
        shmem_team_t &new_team)
{
    new_team = SHMEM_TEAM_INVALID;

    shmem_team *myteam = nullptr;
    myteam = (shmem_team *)calloc(1, sizeof(shmem_team));

    shmem_team *src_team = shmem_team_pool[parent_team];

    int global_pe = src_team->mype;
    int global_PE_start = src_team->start + PE_start * src_team->stride;
    int global_PE_stride = src_team->stride * PE_stride;
    int global_PE_end = global_PE_start + global_PE_stride * (PE_size - 1);

    if (PE_start < 0 || PE_start >= src_team->size || PE_size <= 0 || PE_size > src_team->size || PE_stride < 1) {
        // std::cout << "InValid team create !" << std::endl;                  // TODO LOG
        return -1;
    }

    if (global_PE_start >= shmem_team_pool[0]->size || global_PE_end >= shmem_team_pool[0]->size) {
        // std::cout << "InValid team create !" << std::endl;                  // TODO LOG
        return -1;
    }

    myteam->mype = (global_pe - global_PE_start) / global_PE_stride;

    if (global_pe < global_PE_start || (global_pe - global_PE_start)  % global_PE_stride || myteam->mype >= PE_size) {
        // std::cout << "InValid team create !" << std::endl;                  // TODO LOG
        return -1;
    }

    myteam->start = global_PE_start;
    myteam->stride = global_PE_stride;
    myteam->size = PE_size;

    myteam->team_idx = first_free_idx_fetch();
    if (myteam->team_idx == -1) {
        // std::cout << "EXCEED MAX_TEAM SIZE !!" << std::endl;                  // TODO LOG
        return -1;
    }
    shmem_team_pool[myteam->team_idx] = myteam;

    new_team = myteam->team_idx;
    return 1;
}


int shmem_team_translate_pe(
    shmem_team_t src_team, int src_pe,
    shmem_team_t dest_team)
{
    if (src_team == SHMEM_TEAM_INVALID || dest_team == SHMEM_TEAM_INVALID) return -1;
    shmem_team *src_team_ptr = shmem_team_pool[src_team];
    shmem_team *dest_team_ptr = shmem_team_pool[dest_team];

    if (src_pe > src_team_ptr->size) return -1;

    int global_pe = src_team_ptr->start + src_pe * src_team_ptr->stride;
    int PE_start = dest_team_ptr->start;
    int PE_stride = dest_team_ptr->stride;
    int PE_size = dest_team_ptr->size;

    int n = (global_pe - PE_start) / PE_stride;
    if (global_pe < PE_start || (global_pe - PE_start) % PE_stride || n >= PE_size)
        return -1;
    
    return n;
}


void shmem_team_destroy(shmem_team_t team)
{
    if (team == -1) {
        return;
    }
    pool_avail[team] = 0;
    shmem_team_pool[team] = nullptr;

    return;
}


int shmem_team_finalize() {
    /* Destroy all undestroyed teams*/
    int shmem_max_teams = SHMEM_MAX_TEAMS;
    for (int i = 0; i < shmem_max_teams; i++) {
        if (shmem_team_pool[i] != NULL) shmem_team_destroy((shmem_team_t)i);
    }

    free(shmem_team_pool);

    // shmem_free(shmem_psync_pool);
    // shmem_free(shmem_sync_counter);
    free(pool_avail);
    return 0;
}


int shmem_mype()
{
    return shmem_team_pool[0]->mype;
}


int shmem_n_pes()
{
    return shmem_team_pool[0]->size;
}


int shmem_team_mype(shmem_team_t team)
{
    if (team == SHMEM_TEAM_INVALID)
        return -1;
    else
        return shmem_team_pool[team]->mype;
}


int shmem_team_n_pes(shmem_team_t team)
{
    if (team == SHMEM_TEAM_INVALID)
        return -1;
    else
        return shmem_team_pool[team]->size;
}