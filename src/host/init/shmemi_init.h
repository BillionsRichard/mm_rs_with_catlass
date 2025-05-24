#ifndef SHMEMI_INIT_H
#define SHMEMI_INIT_H

#include "stdint.h"
#include "internal/host_device/shmemi_types.h"

namespace shm {
extern shmemi_device_host_state_t g_state;

int32_t update_device_state(void);

int32_t shmemi_control_barrier_all();

}

#endif // SHMEMI_INIT_H
