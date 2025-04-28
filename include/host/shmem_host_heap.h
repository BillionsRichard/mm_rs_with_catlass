#ifndef SHMEM_HOST_HEAP_H
#define SHMEM_HOST_HEAP_H

void *ShmemMalloc(size_t size);
void ShmemFree(void *ptr);

#endif