#include <iostream>

using namespace std;

#include "init.h"
#include "mem.h"

extern ShmemDeviceHostStateT shmemDeviceHostState;

// ShmemPtr Symmetric?
void* ShmemPtr(void *ptr, int pe)
{
    void *mypePtr = shmemDeviceHostState.p2pHeapBase[ShmemMype()];
    uint64_t offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(mypePtr);
    if (shmemDeviceHostState.heapBase != NULL) {
        return shmemDeviceHostState.heapBase + shmemDeviceHostState.heapSize * pe + offset;
    }
    else {
        return NULL;
    }
}