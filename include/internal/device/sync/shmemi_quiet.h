#ifndef SHEMEI_QUIET_H
#define SHEMEI_QUIET_H

#include "../arch.h"

SHMEM_AICORE_INLINE void ShmemiQuiet() {
    // clear instruction pipes
    pipe_barrier(PIPE_ALL);

    // flush data cache to GM
    DcciEntireCache();
}

#endif