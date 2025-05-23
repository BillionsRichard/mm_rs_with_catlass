#include <iostream>
using namespace std;

#include "shmemi_host_common.h"

// shmem_ptr Symmetric?
void* shmem_ptr(void *ptr, int32_t pe)
{
    uint64_t lower_bound = (uint64_t)shm::gState.p2p_heap_base[shmem_my_pe()];
    uint64_t upper_bound = lower_bound + shm::gState.heap_size;
    if (uint64_t(ptr) < lower_bound || uint64_t(ptr) >= upper_bound) {
        SHM_LOG_ERROR("PE: " << shmem_my_pe() << " Got Ilegal Address !!");
        return nullptr;
    }
    void *mypePtr = shm::gState.p2p_heap_base[shmem_my_pe()];
    uint64_t offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(mypePtr);
    if (shm::gState.heap_base != nullptr) {
        return (void *)((uint64_t)shm::gState.heap_base + shm::gState.heap_size * pe + offset);
    }
    else {
        return nullptr;
    }
}

// Set Memcpy Interfaces necessary UB Buffer.
int32_t shmem_mte_set_ub_params(uint64_t offset, uint32_t ub_size, uint32_t event_id)
{
    shm::gState.mte_config.shmem_ub = offset;
    shm::gState.mte_config.ub_size = ub_size;
    shm::gState.mte_config.event_id = event_id;
    SHMEM_CHECK_RET(shm::UpdateDeviceState());
    return SHMEM_SUCCESS;
}