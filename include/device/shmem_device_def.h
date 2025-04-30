#ifndef SHMEM_DEVICE_DEF_H
#define SHMEM_DEVICE_DEF_H

#include "kernel_operator.h"
#include "host_device/shmem_types.h"

#define SHMEM_GLOBAL __global__ __aicore__
#define SHMEM_DEVICE __attribute__((always_inline)) __aicore__ __inline__

#endif