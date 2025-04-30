#ifndef SHEMEI_QUIET_H
#define SHEMEI_QUIET_H

#include "../shmemi_device_common.h"

SHMEM_DEVICE void ShmemiQuiet() {
    // clear instruction pipes
    AscendC::PipeBarrier<PIPE_ALL>();

    // flush data cache to GM
    DcciEntireCache();
}

#endif