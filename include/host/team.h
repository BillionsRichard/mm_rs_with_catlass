#ifndef SHMEM_TEAM_H
#define SHMEM_TEAM_H

#include <climits>
#include <cstdlib>
#include <cstdbool>

#include "shmem_internal.h"

int ShmemTeamInit(int rank, int size);                    // TODO, No inputs

int ShmemTeamFinalize();

int ShmemTeamSplitStrided(ShmemTeam_t parentTeam, int peStart, int peStride, int peSize, ShmemTeam_t &newTeam);

int ShmemTeamTranslatePE(ShmemTeam_t srcTeam, int srcPe, ShmemTeam_t destTeam);

void ShmemTeamDestroy(ShmemTeam_t team);

int ShmemMype();

int ShmemNpes();

int ShmemTeamMype(ShmemTeam_t team);

int ShmemTeamNpes(ShmemTeam_t team);

#endif