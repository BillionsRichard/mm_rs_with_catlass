#ifndef SHMEM_DEVICE_DEF_H
#define SHMEM_DEVICE_DEF_H

#include "kernel_operator.h"
#include "host_device/shmem_types.h"
/**
 * @addtogroup group_macros
 * @{
*/

/**
 * @private 
*/
#define SHMEM_GLOBAL __global__ __aicore__

/// \def SHMEM_DEVICE
/// \brief A macro that identifies a function on the device side.
#define SHMEM_DEVICE __attribute__((always_inline)) __aicore__ __inline__

/**@} */ // end of group_macros
#endif