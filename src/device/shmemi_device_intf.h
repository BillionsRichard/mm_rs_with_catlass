#ifndef SHMEMI_DEVICE_INTF_H
#define SHMEMI_DEVICE_INTF_H

#include "stdint.h"
#include "host_device/shmem_types.h"

// internal kernels
int32_t shmemi_memset(int32_t *array, int32_t len, int32_t val, int32_t count);

int32_t shmemi_barrier_on_stream(shmem_team_t tid, void *stream);

#endif