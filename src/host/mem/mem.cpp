#include <iostream>

using namespace std;

#include "init_internal.h"
#include "team.h"
#include "mem.h"
#include "data_utils.h"

extern ShmemDeviceHostStateT shmemDeviceHostState;

// ShmemPtr Symmetric?
void* ShmemPtr(void *ptr, int pe)
{
    uint64_t lowerBound = (uint64_t)shmemDeviceHostState.p2pHeapBase[ShmemMype()];
    uint64_t upperBound = lowerBound + shmemDeviceHostState.heapSize;
    if (uint64_t(ptr) < lowerBound || uint64_t(ptr) >= upperBound) {
        std::cout << "PE: " << ShmemMype() << " Got Ilegal Address !!" << std::endl;
    }
    void *mypePtr = shmemDeviceHostState.p2pHeapBase[ShmemMype()];
    uint64_t offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(mypePtr);
    if (shmemDeviceHostState.heapBase != NULL) {
        return shmemDeviceHostState.heapBase + shmemDeviceHostState.heapSize * pe + offset;
    }
    else {
        return NULL;
    }
}

// Set Memcpy Interfaces necessary UB Buffer.
int ShmemSetCopyUB(uint64_t offset, uint32_t ubSize, uint32_t eventID)
{
    int status = SHMEM_SUCCESS;
    shmemDeviceHostState.mteConfig.shmemUB = offset;
    shmemDeviceHostState.mteConfig.ubSize = ubSize;
    shmemDeviceHostState.mteConfig.eventID = eventID;
    CHECK_SHMEM(UpdateDeviceState(), status);

    return status;
}