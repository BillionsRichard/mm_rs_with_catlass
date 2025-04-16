#ifndef _SHMEM_API_H
#define _SHMEM_API_H
#include "types.h"
#include "init.h"

int ShmemInitStatus();

int ShmemInit(int myRank, int nRanks, uint64_t localMemSize);

int ShmemFinalize();

#endif