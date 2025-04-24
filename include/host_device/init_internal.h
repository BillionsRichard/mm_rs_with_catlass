#ifndef _SHMEM_INIT_H
#define _SHMEM_INIT_H

#include "types.h"

int VersionCompatible();

int ShmemOptionsInit();

int ShmemStateInit(ShmemInitAttrT *attributes);

int SmemHeapInit(ShmemInitAttrT *attributes);

int UpdateDeviceState();

int ShmemSetConfig();

int CheckAttr(ShmemInitAttrT *attributes);

#endif
