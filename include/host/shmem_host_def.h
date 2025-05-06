#ifndef SHMEM_HOST_DEF_H
#define SHMEM_HOST_DEF_H
#include <climits>
#include "host_device/shmem_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SHMEM_HOST_API   __attribute__((visibility("default")))

enum shmem_error_code_t : int {
    SHMEM_SUCCESS = 0,
    SHMEM_INVALID_PARAM = -1,
    SHMEM_INVALID_VALUE = -2,
    SHMEM_SMEM_ERROR = -3,
    SHMEM_INNER_ERROR = -4,
    SHMEM_NOT_INITED = -5,
};

enum {
    SHMEM_STATUS_NOT_INITALIZED = 0,
    SHMEM_STATUS_SHM_CREATED,
    SHMEM_STATUS_IS_INITALIZED,
    SHMEM_STATUS_INVALID = INT_MAX,
};

// attr
typedef struct {
    data_op_engine_type_t dataOpEngineType;
    uint32_t shmInitTimeout;
    uint32_t shmCreateTimeout;
    uint32_t controlOperationTimeout;
} shmem_init_optional_attr_t;

typedef struct {
    int version;
    int myRank;
    int nRanks;
    const char* ipPort;
    uint64_t localMemSize;
    shmem_init_optional_attr_t optionAttr;
} shmem_init_attr_t;

#ifdef __cplusplus
}
#endif

#endif