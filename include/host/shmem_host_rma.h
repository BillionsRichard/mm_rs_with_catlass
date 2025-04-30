#ifndef SHMEM_HOST_RMA_H
#define SHMEM_HOST_RMA_H

#ifdef __cplusplus
extern "C" {
#endif

SHMEM_HOST_API void* shmem_ptr(void *ptr, int pe);

SHMEM_HOST_API int shmem_mte_set_ub_params(uint64_t offset, uint32_t ubSize, uint32_t eventID);

#ifdef __cplusplus
}
#endif

#endif