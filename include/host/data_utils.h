#ifndef DATA_UTILS_H
#define DATA_UTILS_H
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "constants.h"

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
        if (status != ACL_ERROR_NONE) {                                                     \
            std::cerr << __FILE__ << ":" << __LINE__ << #x << " return ShmemError: "        \
                    << status << std::endl;                                                 \
            return status;                                                                  \
        }                                                                                   \
    } while (0);

#define CHECK_SHMEM_STATUS(x, status, msg)                                                  \
    do {                                                                                    \
        status = x;                                                                         \
        if (status != ACL_ERROR_NONE) {                                                     \
            ERROR_LOG(msg);                                                                 \
            std::cerr << __FILE__ << ":" << __LINE__ << #x << " return ShmemError: "        \
                    << status << std::endl;                                                 \
            return status;                                                                  \
        }                                                                                   \
    } while (0);

#endif /*DATA_UTILS_H*/