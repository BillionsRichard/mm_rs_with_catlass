#ifndef _SHMEM_INIT_H
#define _SHMEM_INIT_H

class shmem_symmetric_heap;
class shmem_static_heap;

#include "team.h"
#include "shmem_heap.h"

typedef struct {
    /* PE State*/
    int mype;
    int npes;

    shmem_symmetric_heap *heap_obj;
}   shmem_state_t;

extern shmem_state_t *shmem_state;

void shmem_init(int rank, int size);

void shmem_finalize();

void shmem_init_symmetric_heap(shmem_state_t *state, int rank, int size);

void shmem_fini_symmetric_heap(shmem_state_t *state);

#endif
