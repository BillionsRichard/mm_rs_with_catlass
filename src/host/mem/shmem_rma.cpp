#include <iostream>
using namespace std;

#include "shmemi_host_common.h"

// shmem_ptr Symmetric?
void* shmem_ptr(void *ptr, int32_t pe)
{
    uint64_t lowerBound = (uint64_t)shm::gState.p2pHeapBase[shmem_my_pe()];
    uint64_t upperBound = lowerBound + shm::gState.heapSize;
    if (uint64_t(ptr) < lowerBound || uint64_t(ptr) >= upperBound) {
        SHM_LOG_ERROR("PE: " << shmem_my_pe() << " Got Ilegal Address !!");
        return nullptr;
    }
    void *mypePtr = shm::gState.p2pHeapBase[shmem_my_pe()];
    uint64_t offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(mypePtr);
    if (shm::gState.heapBase != nullptr) {
        return (void *)((uint64_t)shm::gState.heapBase + shm::gState.heapSize * pe + offset);
    }
    else {
        return nullptr;
    }
}

// Set Memcpy Interfaces necessary UB Buffer.
int32_t shmem_mte_set_ub_params(uint64_t offset, uint32_t ubSize, uint32_t eventID)
{
    shm::gState.mteConfig.shmemUB = offset;
    shm::gState.mteConfig.ubSize = ubSize;
    shm::gState.mteConfig.eventID = eventID;
    SHMEM_CHECK_RET(shm::UpdateDeviceState());
    return SHMEM_SUCCESS;
}