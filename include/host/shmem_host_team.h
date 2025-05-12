#ifndef SHMEM_HOST_TEAM_H
#define SHMEM_HOST_TEAM_H

#include "host_device/shmem_types.h"

#ifdef __cplusplus
extern "C" {
#endif

SHMEM_HOST_API int shmem_team_split_strided(shmem_team_t parentTeam, int peStart, int peStride, int peSize, shmem_team_t *newTeam);

SHMEM_HOST_API int shmem_team_translate_pe(shmem_team_t srcTeam, int srcPe, shmem_team_t destTeam);

SHMEM_HOST_API void shmem_team_destroy(shmem_team_t team);

SHMEM_HOST_API int shmem_my_pe();

SHMEM_HOST_API int shmem_n_pes();

SHMEM_HOST_API int shmem_team_my_pe(shmem_team_t team);

SHMEM_HOST_API int shmem_team_n_pes(shmem_team_t team);

#ifdef __cplusplus
}
#endif

#endif