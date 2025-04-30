#ifndef SHMEM_HOST_HEAP_H
#define SHMEM_HOST_HEAP_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Allocate <i>size</i> bytes and returns a pointer to the allocated memory. The memory is not initialized.
 *        If <i>size</i> is 0, then <b>shmem_malloc()</b> returns NULL.
 *
 * @param size             [in] bytes to be allocated
 * @return pointer to the allocated memory.
 */
SHMEM_HOST_API void *shmem_malloc(size_t size);

/**
 * @brief Allocate memory for an array of <i>nmemb</i> elements of <i>size</i> bytes each and returns a pointer to the
 *        allocated memory. The memory is set to zero. If <i>nmemb</i> or <i>size</i> is 0, then <b>calloc()</b>
 *        returns either NULL.
 *
 * @param nmemb            [in] elements count
 * @param size             [in] each element size in bytes
 * @return pointer to the allocated memory.
 */
SHMEM_HOST_API void *shmem_calloc(size_t nmemb, size_t size);

/**
 * @brief Allocate <i>size</i> bytes and returns a pointer to the allocated memory. The memory address will be a
 *        multiple of <i>alignment</i>, which must be a power of two.
 *
 * @param alignment        [in] memory address alignment
 * @param size             [in] bytes allocated
 * @return pointer to the allocated memory.
 */
SHMEM_HOST_API void *shmem_align(size_t alignment, size_t size);

/**
 * @brief Free the memory space pointed to by <i>ptr</i>, which must have been returned by a previous call to
 *       <b>shmem_malloc()</b>, <b>calloc()</b>, <b>shmem_align()</b> or <b>realloc()</b>. If <i>ptr</i> is NULL,
 *       no operation is performed.
 * @param ptr              [in] point to memory block to be free.
 */
SHMEM_HOST_API void shmem_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif