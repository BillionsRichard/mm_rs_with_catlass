#ifndef _INIT_INTERNAL_
#define _INIT_INTERNAL_

#include "types.h"

int VersionCompatible();

int ShmemOptionsInit();

int ShmemStateInit(ShmemInitAttrT *attributes);

int SmemHeapInit(ShmemInitAttrT *attributes);

int UpdateDeviceState();

int ShmemSetConfig();

int CheckAttr(ShmemInitAttrT *attributes);

#endif
