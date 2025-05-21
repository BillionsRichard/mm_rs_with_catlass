#ifndef SHMEM_HOST_RMA_H
#define SHMEM_HOST_RMA_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Translate an local symmetric address to remote symmetric address on the specified PE.
 *        Firstly, check whether the input address is legal on local PE. Then translate it into remote address 
 *        on specified PE. Otherwise, returns a null pointer.
 *
 * @param ptr               [in] Symmetric address on local PE.
 * @param pe                [in] The number of the remote PE.
 * @return If the input address is legal, returns a remote symmetric address on the specified PE that can be 
 *         accessed using memory loads and stores. Otherwise, a null pointer is returned.
 */
SHMEM_HOST_API void* shmem_ptr(void *ptr, int pe);

/**
 * @brief Set necessary parameters for put\get.
 *
 * @param offset                [in] The start address on UB.
 * @param ubSize                [in] The Size of Temp UB Buffer.
 * @param eventID               [in] Sync ID for put\get interfaces.
 * @return Returns 0 on success or an error code on failure.
 */
SHMEM_HOST_API int shmem_mte_set_ub_params(uint64_t offset, uint32_t ubSize, uint32_t eventID);

#ifdef __cplusplus
}
#endif

#endif