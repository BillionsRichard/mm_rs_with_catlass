#ifndef SHMEMI_TEAM_H
#define SHMEMI_TEAM_H

#include "stdint.h"

namespace shm {

int32_t shmemi_team_init(int32_t rank, int32_t size);

int32_t shmemi_team_finalize();

}

#endif  // SHMEMI_TEAM_H
