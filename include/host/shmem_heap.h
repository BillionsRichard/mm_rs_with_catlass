#ifndef _SHMEM_HEAP_H
#define _SHMEM_HEAP_H

#include <memory>

void *ShmemMalloc(size_t size);
void ShmemFree(void *ptr);

#endif