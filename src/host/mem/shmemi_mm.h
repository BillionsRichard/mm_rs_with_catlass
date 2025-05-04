#ifndef SHMEMI_MM_H
#define SHMEMI_MM_H

#include "host/shmem_host_def.h"

namespace shm {
int32_t MemoryManagerInitialize(void *base, uint64_t size);
void MemoryManagerDestroy();
}

#endif  // SHMEMI_MM_H
