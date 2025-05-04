#ifndef SHMEMI_INIT_H
#define SHMEMI_INIT_H

#include "stdint.h"
#include "internal/host_device/shmemi_types.h"

namespace shm {
extern ShmemiDeviceHostState gState;

int32_t UpdateDeviceState(void);

int32_t ShmemiControlBarrierAll();

}

#endif // SHMEMI_INIT_H
