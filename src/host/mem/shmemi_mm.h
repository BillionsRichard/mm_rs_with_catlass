#ifndef SHMEMI_MM_H
#define SHMEMI_MM_H

#include "host/shmem_host_def.h"

namespace shm {
int32_t memory_manager_initialize(void *base, uint64_t size);
void memory_manager_destroy();
}

#endif  // SHMEMI_MM_H
