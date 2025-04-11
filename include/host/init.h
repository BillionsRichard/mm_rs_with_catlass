#ifndef _SHMEM_INIT_H
#define _SHMEM_INIT_H

#include "team.h"

typedef struct {
    /* PE State*/
    int mype;
    int npes;

}   ShmemState_t;

extern ShmemState_t *shmemState;

void ShmemInit(int rank, int size);

void ShmemFinalize();

#endif
