#ifndef SHMEMI_HOST_INTF_H
#define SHMEMI_HOST_INTF_H

#include "smem.h"
#include "smem_shm.h"
#include "shmemi_host_def.h"

#include "shmem_host_api.h"

extern ShmemiDeviceHostState gState;

// TODO: not supposed to be exposed
extern ShmemInitAttrT gAttr;

// init
int UpdateDeviceState();

// team
int ShmemiTeamInit(int rank, int size);                    // TODO, No inputs

int ShmemiTeamFinalize();

// internal kernels
void ShmemiMemset(int* array, int len, int val);

#endif
