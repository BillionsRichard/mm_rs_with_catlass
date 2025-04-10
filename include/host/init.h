#ifndef _SHMEM_INIT_H
#define _SHMEM_INIT_H

#include "team.h"

typedef struct {
    /* PE State*/
    int mype;
    int npes;

}   shmem_state_t;

extern shmem_state_t *shmem_state;

void shmem_init(int rank, int size);

void shmem_finalize();

#endif
