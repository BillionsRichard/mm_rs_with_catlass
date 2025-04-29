#ifndef SHMEM_HOST_RMA_H
#define SHMEM_HOST_RMA_H

void* ShmemPtr(void *ptr, int pe);

int ShmemSetCopyUB(uint64_t offset, uint32_t ubSize, uint32_t eventID);

#endif