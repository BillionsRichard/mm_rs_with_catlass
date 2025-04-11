#ifndef _SHMEM_API_H
#define _SHMEM_API_H
#include "types.h"
#include "init.h"

int ShmemInitStatus();

int ShmemInit(int rank, int size);

int ShmemInit(uint32_t flag, ShmemInitAttrT *attributes, smem_shm_t &handle);

int ShmemFinalize(smem_shm_t handle, uint32_t flag);

#endif