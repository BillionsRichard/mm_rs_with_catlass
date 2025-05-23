#ifndef SHMEM_HOST_DEF_H
#define SHMEM_HOST_DEF_H
#include <climits>
#include "host_device/shmem_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup group_macros Macros
 * @{
*/
/// \def SHMEM_HOST_API
/// \brief A macro that identifies a function on the host side.
#define SHMEM_HOST_API   __attribute__((visibility("default")))
/**@} */ // end of group_macros

/**
 * @defgroup group_enums Enumerations
 * @{
*/

/**
 * @brief Error code for the SHMEM library.
*/
enum shmem_error_code_t : int {
    SHMEM_SUCCESS = 0,          ///< Task execution was successful.
    SHMEM_INVALID_PARAM = -1,   ///< There is a problem with the parameters.
    SHMEM_INVALID_VALUE = -2,   ///< There is a problem with the range of the value of the parameter.
    SHMEM_SMEM_ERROR = -3,      ///< There is a problem with SMEM.
    SHMEM_INNER_ERROR = -4,     ///< This is a problem caused by an internal error.
    SHMEM_NOT_INITED = -5,      ///< This is a problem caused by an uninitialization.
};

/**
 * @brief The state of the SHMEM library initialization.
*/
enum shmem_init_status_t{
    SHMEM_STATUS_NOT_INITALIZED = 0,    ///< Uninitialized.
    SHMEM_STATUS_SHM_CREATED,           ///< Shared memory heap creation is complete.
    SHMEM_STATUS_IS_INITALIZED,         ///< Initialization is complete.
    SHMEM_STATUS_INVALID = INT_MAX,     ///< Invalid status code.
};

/**@} */ // end of group_enums

/**
 * @defgroup group_structs Structs
 * @{
*/

/**
 * @struct shmem_init_optional_attr_t
 * @brief Optional parameter for the attributes used for initialization.
 *
 * - int version: version
 * - data_op_engine_type_t dataOpEngineType: dataOpEngineType
 * - uint32_t shmInitTimeout: shmInitTimeout
 * - uint32_t shmCreateTimeout: shmCreateTimeout
 * - uint32_t controlOperationTimeout: controlOperationTimeout
*/
typedef struct {
    int version;
    data_op_engine_type_t dataOpEngineType;
    uint32_t shmInitTimeout;
    uint32_t shmCreateTimeout;
    uint32_t controlOperationTimeout;
} shmem_init_optional_attr_t;

/**
 * @struct shmem_init_attr_t
 * @brief Mandatory parameter for attributes used for initialization.
 *
 * - int myRank: The rank of the current process.
 * - int nRanks: The total rank number of all processes.
 * - const char* ipPort: The ip and port of the communication server. The port must not conflict with other modules and processes.
 * - uint64_t localMemSize: The size of shared memory currently occupied by current rank.
 * - shmem_init_optional_attr_t optionAttr: Optional Parameters.
*/
typedef struct {
    int myRank;      
    int nRanks;  
    const char* ipPort; 
    uint64_t localMemSize; 
    shmem_init_optional_attr_t optionAttr;  
} shmem_init_attr_t;

/**@} */ // end of group_structs

#ifdef __cplusplus
}
#endif

#endif