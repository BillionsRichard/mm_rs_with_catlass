#ifndef _SHMEM_INIT_H
#define _SHMEM_INIT_H

#include "team.h"
#include "mem.h"
#include "types.h"

int VersionCompatible();

int ShmemOptionsInit();

ShmemInitAttr CreateAttributes(int myRank, int nRanks, uint64_t localMemSize);

int CommAttrInit(ShmemInitAttr *shmemInitAttr);

int ShmemStateInit(ShmemInitAttrT *attributes);

int SmemHeapInit(ShmemInitAttrT *attributes);

int UpdateDeviceState();

int ShmemTeamInit(ShmemInitAttrT *attributes);

int ShmemHostInitAttr(ShmemInitAttrT *attributes);

int ShmemSetConfig();

int CheckAttr(int myRank, int nRanks, uint64_t localMemSize);

#endif
