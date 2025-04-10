cmake_minimum_required(VERSION 3.19)
project(SHMEM)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 生成位置无关代码
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 设置可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 设置安装路径
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)

# 获取CANN相关环境变量
if (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

IF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    SET(ARCH aarch64)
ELSEIF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    SET(ARCH x86_64)
ENDIF ()

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

# 添加子目录
add_subdirectory(src/host)
add_subdirectory(src/device)
add_subdirectory(tests)

message("== ASCEND_HOME_PATH : ${ASCEND_HOME_PATH}")
message("== ASCEND_DRIVER_PATH : ${ASCEND_DRIVER_PATH}")