#ifndef SHMEM_TYPES_H
#define SHMEM_TYPES_H

#include <vector>
#include "stdint.h"
#include "limits.h"

#define DEFAULT_TIMEOUT 120
#define ATTR_SCALAR_INVALID -1

enum DataOpEngineType {
    SHMEM_DATA_OP_MTE = 0x01,
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
extern ShmemInitAttrT shmemInitAttr;


#endif /*SHMEM_TYPES_H*/