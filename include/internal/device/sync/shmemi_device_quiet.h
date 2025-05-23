#ifndef SHEMEI_QUIET_H
#define SHEMEI_QUIET_H

#include "internal/device/shmemi_device_common.h"

SHMEM_DEVICE void shmemi_quiet() {
    // clear instruction pipes
    AscendC::PipeBarrier<PIPE_ALL>();

    // flush data cache to GM
    dcci_entire_cache();
}

#endif