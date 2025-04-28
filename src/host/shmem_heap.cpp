//
// Created by l00915220 on 2025/4/18.
//
#include <mutex>

#include "shmemi_host_intf.h"
#include "shmemi_mspace.h"

namespace {
bool initialized = false;
std::mutex shmMutex;
Mspace space;
}

void *ShmemMalloc(size_t size)
{
    std::unique_lock<std::mutex> lockGuard{ shmMutex };
    if (!initialized) {
        space.AddNewChunk(gState.heapBase, gState.heapSize - SHMEM_EXTRA_SIZE);
        initialized = true;
    }

    return space.alloc(size);
}

void ShmemFree(void *ptr)
{
    std::unique_lock<std::mutex> lockGuard{ shmMutex };
    return space.free(ptr);
}