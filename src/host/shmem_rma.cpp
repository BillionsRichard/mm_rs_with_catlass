#include <iostream>

using namespace std;

#include "shmemi_host_intf.h"

// ShmemPtr Symmetric?
void* ShmemPtr(void *ptr, int pe)
{
    uint64_t lowerBound = (uint64_t)gState.p2pHeapBase[ShmemMype()];
    uint64_t upperBound = lowerBound + gState.heapSize;
    if (uint64_t(ptr) < lowerBound || uint64_t(ptr) >= upperBound) {
        std::cout << "PE: " << ShmemMype() << " Got Ilegal Address !!" << std::endl;
    }
    void *mypePtr = gState.p2pHeapBase[ShmemMype()];
    uint64_t offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(mypePtr);
    if (gState.heapBase != NULL) {
        return gState.heapBase + gState.heapSize * pe + offset;
    }
    else {
        return NULL;
    }
}

// Set Memcpy Interfaces necessary UB Buffer.
int ShmemSetCopyUB(uint64_t offset, uint32_t ubSize, uint32_t eventID)
{
    int status = SHMEM_SUCCESS;
    gState.mteConfig.shmemUB = offset;
    gState.mteConfig.ubSize = ubSize;
    gState.mteConfig.eventID = eventID;
    CHECK_SHMEM(UpdateDeviceState(), status);

    return status;
}