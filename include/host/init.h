#ifndef _SHMEM_INIT_H
#define _SHMEM_INIT_H

#include "team.h"
#include "types.h"

int VersionCompatible();

int ShmemOptionsInit();

ShmemInitAttr CreateAttributes(int id, const char* ipPort, int myRank, int nRanks, int deviceId,
                                uint64_t localMemSize =2097152, uint64_t extraSize = 0, 
                                smem_shm_data_op_type dataOpType = SMEMS_DATA_OP_MTE,
                                int timeout = 30);

int ShmemStateInit(ShmemInitAttrT *attributes);

int SmemHeapInit(uint32_t flag, ShmemInitAttrT *attributes);

int UpdateDeviceState();

int ShmemTeamInit(uint32_t flag, ShmemInitAttrT *attributes);

int ShmemHostInitAttr(uint32_t flag, ShmemInitAttrT *attributes);

int ShmemSetConfig();

ShmemInitAttr CreateAttributes();

#endif
