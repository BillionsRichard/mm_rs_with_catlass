#ifndef SHMEM_HOST_DEF_H
#define SHMEM_HOST_DEF_H
#include <climits>
#include <iostream>

#include "host_device/shmem_types.h"

enum Status : int {
    SHMEM_SUCCESS = 0,
    ERROR_INVALID_PARAM,
    ERROR_INVALID_VALUE,
    ERROR_SMEM_ERROR
};

enum {
    SHMEM_STATUS_NOT_INITALIZED = 0,
    SHMEM_STATUS_SHM_CREATED,
    SHMEM_STATUS_IS_INITALIZED,
    SHMEM_STATUS_INVALID = INT_MAX,
};

// attr
typedef struct {
    DataOpEngineType dataOpEngineType;
    uint32_t shmInitTimeout;
    uint32_t shmCreateTimeout;
    uint32_t controlOperationTimeout;
} ShmemInitOptionalAttr;

typedef struct {
    int version;
    int myRank;
    int nRanks;
    const char* ipPort;
    uint64_t localMemSize;
    ShmemInitOptionalAttr optionAttr;
} ShmemInitAttrT;

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO] " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN] " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR] " fmt "\n", ##args)

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

#define CHECK_ACL_RET(x, msg)                                                               \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << msg << ":" << " aclError:" << __ret << std::endl;                  \
            return __ret;                                                                   \
        }                                                                                   \
    } while (0);

#define CHECK_SHMEM(x, status)                                                              \
    do {                                                                                    \
        status = x;                                                                         \
        if (status != SHMEM_SUCCESS) {                                                     \
            std::cerr << __FILE__ << ":" << __LINE__ << #x << " return ShmemError: "        \
                    << status << std::endl;                                                 \
            return status;                                                                  \
        }                                                                                   \
    } while (0);

#define CHECK_SHMEM_STATUS(x, status, msg)                                                  \
    do {                                                                                    \
        status = x;                                                                         \
        if (status != SHMEM_SUCCESS) {                                                     \
            ERROR_LOG(msg);                                                                 \
            std::cerr << __FILE__ << ":" << __LINE__ << #x << " return ShmemError: "        \
                    << status << std::endl;                                                 \
            return status;                                                                  \
        }                                                                                   \
    } while (0);

#endif