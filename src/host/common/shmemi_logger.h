/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */
#ifndef SHMEM_SHM_OUT_LOGGER_H
#define SHMEM_SHM_OUT_LOGGER_H

#include <ctime>
#include <climits>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <unistd.h>
#include <sstream>
#include <sys/time.h>
#include <sys/syscall.h>

namespace shm {
using external_log = void (*)(int32_t, const char *);

enum log_level : int32_t {
    DEBUG_LEVEL = 0,
    INFO_LEVEL,
    WARN_LEVEL,
    ERROR_LEVEL,
    BUTT_LEVEL /* no use */
};

class shm_out_logger {
public:
    static shm_out_logger &Instance()
    {
        static shm_out_logger g_logger;
        return g_logger;
    }

    inline void set_log_level(log_level level)
    {
        m_log_level = level;
    }

    inline void set_extern_log_func(external_log func, bool force_update = false)
    {
        if (m_log_func == nullptr || force_update) {
            m_log_func = func;
        }
    }

    inline void log(int32_t level, const std::ostringstream &oss)
    {
        if (m_log_func != nullptr) {
            m_log_func(level, oss.str().c_str());
            return;
        }

        if (level < m_log_level) {
            return;
        }

        struct timeval tv {};
        char str_time[24];

        gettimeofday(&tv, nullptr);
        time_t time_stamp = tv.tv_sec;
        struct tm local_time {};
        if (strftime(str_time, sizeof str_time, "%Y-%m-%d %H:%M:%S.", localtime_r(&time_stamp, &local_time)) != 0) {
            std::cout << str_time << std::setw(6) << std::setfill('0') << tv.tv_usec << " " << log_level_desc(level) << " "
                      << syscall(SYS_gettid) << " " << oss.str() << std::endl;
        } else {
            std::cout << " Invalid time " << log_level_desc(level) << " " << syscall(SYS_gettid) << " " << oss.str()
                      << std::endl;
        }
    }

    shm_out_logger(const shm_out_logger &) = delete;
    shm_out_logger(shm_out_logger &&) = delete;

    ~shm_out_logger()
    {
        m_log_func = nullptr;
    }

private:
    shm_out_logger() = default;

    inline const std::string &log_level_desc(int32_t level)
    {
        static std::string invalid = "invalid";
        if (level < DEBUG_LEVEL || level >= BUTT_LEVEL) {
            return invalid;
        }
        return m_log_level_desc[level];
    }

private:
    const std::string m_log_level_desc[BUTT_LEVEL] = {"debug", "info", "warn", "error"};

    log_level m_log_level = INFO_LEVEL;
    external_log m_log_func = nullptr;
};
}  // namespace shm

#ifndef SHM_LOG_FILENAME_SHORT
#define SHM_LOG_FILENAME_SHORT (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif
#define SHM_OUT_LOG(LEVEL, ARGS)                                                         \
    do {                                                                                 \
        std::ostringstream oss;                                                          \
        oss << "[SHMEM " << SHM_LOG_FILENAME_SHORT << ":" << __LINE__ << "] " << ARGS;   \
        shm::shm_out_logger::Instance().log(LEVEL, oss);                                 \
    } while (0)

#define SHM_LOG_DEBUG(ARGS) SHM_OUT_LOG(shm::DEBUG_LEVEL, ARGS)
#define SHM_LOG_INFO(ARGS) SHM_OUT_LOG(shm::INFO_LEVEL, ARGS)
#define SHM_LOG_WARN(ARGS) SHM_OUT_LOG(shm::WARN_LEVEL, ARGS)
#define SHM_LOG_ERROR(ARGS) SHM_OUT_LOG(shm::ERROR_LEVEL, ARGS)

#define SHM_ASSERT_RETURN(ARGS, RET)             \
    do {                                         \
        if (__builtin_expect(!(ARGS), 0) != 0) { \
            SHM_LOG_ERROR("Assert " << #ARGS);   \
            return RET;                          \
        }                                        \
    } while (0)

#define SHM_ASSERT_RET_VOID(ARGS)                \
    do {                                         \
        if (__builtin_expect(!(ARGS), 0) != 0) { \
            SHM_LOG_ERROR("Assert " << #ARGS);   \
            return;                              \
        }                                        \
    } while (0)

#define SHM_ASSERT_RETURN_NOLOG(ARGS, RET)       \
    do {                                         \
        if (__builtin_expect(!(ARGS), 0) != 0) { \
            return RET;                          \
        }                                        \
    } while (0)

#define SHM_ASSERT(ARGS)                         \
    do {                                         \
        if (__builtin_expect(!(ARGS), 0) != 0) { \
            SHM_LOG_ERROR("Assert " << #ARGS);   \
        }                                        \
    } while (0)

#define SHMEM_CHECK_RET(x)                                       \
    do {                                                         \
        int32_t check_ret = x;                                    \
        if (check_ret != 0) {                                     \
            SHM_LOG_ERROR(" return shmem error: " << check_ret);   \
            return check_ret;                                     \
        }                                                        \
    } while (0);


#endif  //SHMEM_SHM_OUT_LOGGER_H
