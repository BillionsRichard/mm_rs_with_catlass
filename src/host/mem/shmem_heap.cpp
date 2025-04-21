//
// Created by l00915220 on 2025/4/18.
//
#include <mutex>

#include "types_internal.h"
#include "mspace.h"
#include "shmem_heap.h"

namespace {
bool initialized = false;
std::mutex shmMutex;
Mspace space;
}

void *ShmemMalloc(size_t size)
{
    std::unique_lock<std::mutex> lockGuard{ shmMutex };
    if (!initialized) {
        space.AddNewChunk(shmemDeviceHostState.heapBase, shmemDeviceHostState.heapSize - DEFAULT_EXTRA_SIZE);
        initialized = true;
    }

    return space.alloc(size);
}

void ShmemFree(void *ptr)
{
    std::unique_lock<std::mutex> lockGuard{ shmMutex };
    return space.free(ptr);
}