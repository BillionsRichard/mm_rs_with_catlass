#ifndef SHMEM_TYPES_H
#define SHMEM_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @addtogroup group_enums
 * @{
*/

/**
 * @brief Team's index.
*/
enum shmem_team_index_t{
    SHMEM_TEAM_INVALID = -1,
    SHMEM_TEAM_WORLD = 0
};

/**
 * @brief Data op engine type.
*/
enum data_op_engine_type_t {
    SHMEM_DATA_OP_MTE = 0x01,
};

/**@} */ // end of group_enums

/**
 * @defgroup group_typedef Typedef
 * @{

*/
/**
 * @brief A typedef of int
*/
typedef int shmem_team_t;

/**@} */ // end of group_typedef

#ifdef __cplusplus
}
#endif

#endif /*SHMEM_TYPES_H*/