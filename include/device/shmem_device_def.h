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

// Non-Contiguous Datacopy Param
struct non_contiguous_copy_param
{
    uint32_t repeat;
    uint32_t length;
    uint32_t src_ld;     // src data leading dimension. Interval between the head of the repeat and the head of the following repeat
    uint32_t dst_ld;     // dst data leading dimension
};

/**@} */ // end of group_macros
#endif