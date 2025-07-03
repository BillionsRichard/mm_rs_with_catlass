#include <iostream>
#include "shmemi_host_common.h"

using namespace std;
// shmem_ptr Symmetric?
void* shmem_ptr(void *ptr, int32_t pe)
{
    uint64_t lower_bound = (uint64_t)shm::g_state.p2p_heap_base[shmem_my_pe()];
    uint64_t upper_bound = lower_bound + shm::g_state.heap_size;
    if (uint64_t(ptr) < lower_bound || uint64_t(ptr) >= upper_bound) {
        SHM_LOG_ERROR("PE: " << shmem_my_pe() << " Got Ilegal Address !!");
        return nullptr;
    }
    void *mype_ptr = shm::g_state.p2p_heap_base[shmem_my_pe()];
    uint64_t offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(mype_ptr);
    if (shm::g_state.heap_base != nullptr) {
        return (void *)((uint64_t)shm::g_state.heap_base + shm::g_state.heap_size * pe + offset);
    }
    else {
        return nullptr;
    }
}

// Set Memcpy Interfaces necessary UB Buffer.
int32_t shmem_mte_set_ub_params(uint64_t offset, uint32_t ub_size, uint32_t event_id)
{
    shm::g_state.mte_config.shmem_ub = offset;
    shm::g_state.mte_config.ub_size = ub_size;
    shm::g_state.mte_config.event_id = event_id;
    SHMEM_CHECK_RET(shm::update_device_state());
    return SHMEM_SUCCESS;
}