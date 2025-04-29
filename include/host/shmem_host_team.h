#ifndef SHMEM_HOST_TEAM_H
#define SHMEM_HOST_TEAM_H

#include "host_device/shmem_types.h"

int ShmemTeamSplitStrided(ShmemTeam parentTeam, int peStart, int peStride, int peSize, ShmemTeam &newTeam);

int ShmemTeamTranslatePE(ShmemTeam srcTeam, int srcPe, ShmemTeam destTeam);

void ShmemTeamDestroy(ShmemTeam team);

int ShmemMype();

int ShmemNpes();

int ShmemTeamMype(ShmemTeam team);

int ShmemTeamNpes(ShmemTeam team);

#endif