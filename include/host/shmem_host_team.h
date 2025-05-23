#ifndef SHMEM_HOST_TEAM_H
#define SHMEM_HOST_TEAM_H

#include "host_device/shmem_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Collective Interface. Creates a new SHMEM team from an existing parent team.
 * 
 * @param parent_team        [in] A team handle.
 * @param pe_start           [in] The first PE number of the subset of PEs from the parent team.
 * @param pe_stride          [in] The stride between team PE numbers in the parent team.
 * @param pe_size            [in] The total number of PEs in new team.
 * @param new_team           [out] A team handle.
 *
 * @return 0 on successful creation of new_team; otherwise nonzero.
 */
SHMEM_HOST_API int shmem_team_split_strided(shmem_team_t parent_team, int pe_start, int pe_stride, int pe_size, shmem_team_t *new_team);

/**
 * @brief Translate a given PE number in one team into the corresponding PE number in another team.
 * 
 * @param src_team           [in] A team handle.
 * @param src_pe             [in] The PE number in src_team.
 * @param dest_team          [in] A team handle.
 *
 * @return The number of PEs in the specified team. 
 *         If the team handle is NVSHMEM_TEAM_INVALID, returns -1.
 */
SHMEM_HOST_API int shmem_team_translate_pe(shmem_team_t src_team, int src_pe, shmem_team_t dest_team);

/**
 * @brief Collective Interface. Destroys the team referenced by the team handle.
 * 
 * @param team              [in] A team handle.
 */
SHMEM_HOST_API void shmem_team_destroy(shmem_team_t team);

/**
 * @brief Returns the PE number of the local PE
 *
 * @return Integer between 0 and npes - 1
 */
SHMEM_HOST_API int shmem_my_pe();

/**
 * @brief Returns the number of PEs running in the program.
 *
 * @return Number of PEs in the program.
 */
SHMEM_HOST_API int shmem_n_pes();

/**
 * @brief Returns the number of the calling PE in the specified team.
 * 
 * @param team              [in] A team handle.
 *
 * @return The number of the calling PE within the specified team. 
 *         If the team handle is NVSHMEM_TEAM_INVALID, returns -1.
 */
SHMEM_HOST_API int shmem_team_my_pe(shmem_team_t team);

/**
 * @brief Returns the number of PEs in the team.
 * 
 * @param team              [in] A team handle.
 *
 * @return The number of PEs in the specified team. 
 *         If the team handle is NVSHMEM_TEAM_INVALID, returns -1.
 */
SHMEM_HOST_API int shmem_team_n_pes(shmem_team_t team);

#ifdef __cplusplus
}
#endif

#endif