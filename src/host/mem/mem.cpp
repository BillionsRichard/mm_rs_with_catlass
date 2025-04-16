#include <iostream>

using namespace std;

#include "init.h"
#include "mem.h"

extern ShmemDeviceHostStateT shmemDeviceHostState;

// ShmemPtr Symmetric?
void* ShmemPtr(void *ptr, int pe)
{
    if (shmemDeviceHostState.heapBase != NULL) {
        return shmemDeviceHostState.heapBase + shmemDeviceHostState.heapSize * pe;
    }
    else {
        return NULL;
    }
}