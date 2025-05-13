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
 *        Default data operation engine type: SHMEM_DATA_OP_MTE
 *
 * @param attributes Pointer to the attributes to modify the data operation engine type
 * @param value Value of data operation engine type
 * @return Returns 0 on success or an error code on failure
 */
SHMEM_HOST_API int shmem_set_data_op_engine_type(shmem_init_attr_t *attributes, data_op_engine_type_t value);

/**
 * @brief Modify the timeout in the attributes that will be used for initialization.
 *        Default timeout: 120
 *
 * @param attributes Pointer to the attributes to modify the data operation engine type
 * @param value Value of timeout
 * @return Returns 0 on success or an error code on failure
 */
SHMEM_HOST_API int shmem_set_timeout(shmem_init_attr_t *attributes, uint32_t value);

/**
 * @brief Initialization based on user-defined attributes.
 *        The default attributes is automatically used when the value is a null pointer.
 *
 * @param attributes Pointer to the user-defined attributes.
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