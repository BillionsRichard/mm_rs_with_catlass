#include <memory>
#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "shmemi_mm_heap.h"

namespace shm {
namespace {
std::shared_ptr<MemoryHeap> shmMemoryHeap;
}

int32_t MemoryManagerInitialize(void *base, uint64_t size)
{
    shmMemoryHeap = std::make_shared<MemoryHeap>(base, size);
    if (shmMemoryHeap == nullptr) {
        return SHMEM_INNER_ERROR;
    }
    return SHMEM_SUCCESS;
}

void MemoryManagerDestroy()
{
    shmMemoryHeap.reset();
}
} // namespace shm

void *shmem_malloc(size_t size)
{
    if (shm::shmMemoryHeap == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return nullptr;
    }

    void *ptr = shm::shmMemoryHeap->Allocate(size);
    SHM_LOG_DEBUG("shmem_malloc(" << size << ") = " << ptr);
    auto ret = shm::ShmemiControlBarrierAll();
    if (ret != 0) {
        SHM_LOG_ERROR("malloc mem barrier failed, ret: " << ret);
        shm::shmMemoryHeap->Release(ptr);
        ptr = nullptr;
    }
    return ptr;
}

void *shmem_calloc(size_t nmemb, size_t size)
{
    if (shm::shmMemoryHeap == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return nullptr;
    }

    auto totalSize = nmemb * size;
    auto ptr = shm::shmMemoryHeap->Allocate(totalSize);
    if (ptr != nullptr) {
        auto ret = aclrtMemset(ptr, size, 0, size);
        if (ret != 0) {
            SHM_LOG_ERROR("shmem_calloc(" << nmemb << ", " << size << ") memset failed: " << ret);
            shm::shmMemoryHeap->Release(ptr);
            ptr = nullptr;
        }
    }

    auto ret = shm::ShmemiControlBarrierAll();
    if (ret != 0) {
        SHM_LOG_ERROR("calloc mem barrier failed, ret: " << ret);
        shm::shmMemoryHeap->Release(ptr);
        ptr = nullptr;
    }

    SHM_LOG_DEBUG("shmem_calloc(" << nmemb << ", " << size << ") = " << ptr);
    return ptr;
}

void *shmem_align(size_t alignment, size_t size)
{
    if (shm::shmMemoryHeap == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return nullptr;
    }

    auto ptr = shm::shmMemoryHeap->AlignedAllocate(alignment, size);
    auto ret = shm::ShmemiControlBarrierAll();
    if (ret != 0) {
        SHM_LOG_ERROR("shmem_align barrier failed, ret: " << ret);
        shm::shmMemoryHeap->Release(ptr);
        ptr = nullptr;
    }
    SHM_LOG_DEBUG("shmem_align(" << alignment << ", " << size << ") = " << ptr);
    return ptr;
}

void shmem_free(void *ptr)
{
    if (shm::shmMemoryHeap == nullptr) {
        SHM_LOG_ERROR("Memory Heap Not Initialized.");
        return;
    }

    auto ret = shm::shmMemoryHeap->Release(ptr);
    if (ret != 0) {
        SHM_LOG_ERROR("Release for " << ptr << " failed: " << ret);
    }

    SHM_LOG_DEBUG("shmem_free(" << ptr << ") = " << ret);
}