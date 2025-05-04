#ifndef SHMEMI_TEAM_H
#define SHMEMI_TEAM_H

#include "stdint.h"

namespace shm {

int32_t ShmemiTeamInit(int32_t rank, int32_t size);

int32_t ShmemiTeamFinalize();

}

#endif  // SHMEMI_TEAM_H
