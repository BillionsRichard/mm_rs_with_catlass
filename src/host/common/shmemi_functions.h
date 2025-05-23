/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */
#ifndef SHMEM_SHM_FUNCTION_H
#define SHMEM_SHM_FUNCTION_H

#include <cstdint>
#include "shmemi_logger.h"

namespace shm {
class Func {
public:
    /**
     * @brief Get real path
     *
     * @param path         [in/out] input path, converted realpath
     * @return true if successful
     */
    static bool get_real_path(std::string &path);

    /**
     * @brief Get real path of a library and check if exists
     *
     * @param lib_dir_path   [in] dir path of the library
     * @param lib_name      [in] library name
     * @param real_path     [out] realpath of the library
     * @return true if successful
     */
    static bool get_library_real_path(const std::string &lib_dir_path, const std::string &lib_name, std::string &real_path);
};

inline bool Func::get_real_path(std::string &path)
{
    if (path.empty() || path.size() > PATH_MAX) {
        SHM_LOG_ERROR("Failed to get realpath of [" << path << "] as path is invalid");
        return false;
    }

    /* It will allocate memory to store path */
    char *real_path = realpath(path.c_str(), nullptr);
    if (real_path == nullptr) {
        SHM_LOG_ERROR("Failed to get realpath of [" << path << "] as error " << errno);
        return false;
    }

    path = real_path;
    free(real_path);
    real_path = nullptr;
    return true;
}

inline bool Func::get_library_real_path(const std::string &lib_dir_path, const std::string &lib_name, std::string &real_path)
{
    std::string tmpFullPath = lib_dir_path;
    if (!get_real_path(tmpFullPath)) {
        return false;
    }

    if (tmpFullPath.back() != '/') {
        tmpFullPath.push_back('/');
    }

    tmpFullPath.append(lib_name);
    auto ret = ::access(tmpFullPath.c_str(), F_OK);
    if (ret != 0) {
        SHM_LOG_ERROR(tmpFullPath << " cannot be accessed, ret: " << ret);
        return false;
    }

    real_path = tmpFullPath;
    return true;
}

#define DL_LOAD_SYM(TARGET_FUNC_VAR, TARGET_FUNC_TYPE, FILE_HANDLE, SYMBOL_NAME)           \
    do {                                                                                   \
        TARGET_FUNC_VAR = (TARGET_FUNC_TYPE)dlsym(FILE_HANDLE, SYMBOL_NAME);               \
        if (TARGET_FUNC_VAR == nullptr) {                                                  \
            SHM_LOG_ERROR("Failed to call dlsym to load SYMBOL_NAME, error" << dlerror()); \
            dlclose(FILE_HANDLE);                                                          \
            return false;                                                                  \
        }                                                                                  \
    } while (0)
}  // namespace shm

#endif  //SHMEM_SHM_FUNCTION_H
