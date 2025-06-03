#ifndef SHMEM_DEVICE_DEF_H
#define SHMEM_DEVICE_DEF_H

#include "kernel_operator.h"
#include "host_device/shmem_types.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @addtogroup group_macros
 * @{
*/

// Non-Contiguous Datacopy Param
struct non_contiguous_copy_param
{
    uint32_t repeat;
    uint32_t length;
    uint32_t src_ld;     // src data leading dimension. Interval between the head of the repeat and the head of the following repeat
    uint32_t dst_ld;     // dst data leading dimension
};

/**@} */ // end of group_macros

#ifdef __cplusplus
}
#endif

#endif