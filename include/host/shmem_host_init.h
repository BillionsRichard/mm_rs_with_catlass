#ifndef SHMEM_HOST_INIT_H
#define SHMEM_HOST_INIT_H

#include "host_device/shmem_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Query the current initialization status.
 *
 * @param 
 * @return Returns initialization status. Returning SHMEM_STATUS_IS_INITALIZED indicates that initialization is complete.
 */
SHMEM_HOST_API int shmem_init_status();

/**
 * @brief Set the default attributes to be used in <b>shmem_init</b>.
 * @param myRank            [in] Current rank
 * @param nRanks            [in] Total number of ranks
 * @param localMemSize      [in] The size of shared memory currently occupied by current rank
 * @param ipPort            [in] The ip and port number of the sever, e.g. tcp://ip:port
 * @param attributes        [out] Pointer to the default attributes used for initialization
 * @return Returns 0 on success or an error code on failure
 */
SHMEM_HOST_API int shmem_set_attr(int myRank, int nRanks, uint64_t localMemSize, const char* ipPort, shmem_init_attr_t **attributes);

/**
 * @brief Modify the data operation engine type in the attributes that will be used for initialization.
 *        If this method is not used, the default dataOpEngineType value is SHMEM_DATA_OP_MTE
 *        if method <b>shmem_set_attr</b> is used after this method, the dataOpEngineType param will be overwritten by the default value.
 *
 * @param attributes        [in/out] Pointer to the attributes to modify the data operation engine type
 * @param value             [in] Value of data operation engine type
 * @return Returns 0 on success or an error code on failure
 */
SHMEM_HOST_API int shmem_set_data_op_engine_type(shmem_init_attr_t *attributes, data_op_engine_type_t value);

/**
 * @brief Modify the timeout in the attributes that will be used for initialization.
 *        If this method is not used, the default timeout value is 120
 *        if method <b>shmem_set_attr</b> is used after this method, the timeout param will be overwritten by the default value.
 *
 * @param attributes        [in/out] Pointer to the attributes to modify the data operation engine type
 * @param value             [in] Value of timeout
 * @return Returns 0 on success or an error code on failure
 */
SHMEM_HOST_API int shmem_set_timeout(shmem_init_attr_t *attributes, uint32_t value);

/**
 * @brief Initialization based on attributes and build the shmem library.
 *        Attributes can be created by users or obtained by calling <b>shmem_set_attr</b>.
 *        The default attributes is automatically used when the attributes value is a null pointer.
 *
 * @param attributes        [in] Pointer to the user-defined attributes.
 * @return Returns 0 on success or an error code on failure
 */
SHMEM_HOST_API int shmem_init(shmem_init_attr_t *attributes = nullptr);

/**
 * @brief Ends the program previously initialized by <b>shmem_init</b>.
 *        Release all resources used by the SHMEM library.
 *
 * @param 
 * @return Returns 0 on success or an error code on failure
 */
SHMEM_HOST_API int shmem_finalize();

#ifdef __cplusplus
}
#endif

#endif