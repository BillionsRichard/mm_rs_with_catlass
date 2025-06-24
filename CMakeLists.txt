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
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install/shmem)

# 获取CANN相关环境变量
if (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

IF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    SET(ARCH aarch64)
ELSEIF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    SET(ARCH x86_64)
ENDIF ()

option(USE_UNIT_TEST "USE_UNIT_TEST" OFF)

message(STATUS "USE_UNIT_TEST:${USE_UNIT_TEST}")

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

# shmem库编译-device能力
set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend910B3" CACHE STRING "system on chip type")

# 遗留问题: 此处一定要选Debug?
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()

set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}" CACHE STRING "ASCEND_CANN_PACKAGE_PATH")
set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
set(SHMEM_SOURCE_ROOT ${PROJECT_SOURCE_DIR})
add_link_options(-Wl,-z,relro,-z,now)
# 添加子目录
add_subdirectory(src/host)
add_subdirectory(src/device)

if(USE_UNIT_TEST)
    add_subdirectory(tests)
endif()

message("== ASCEND_HOME_PATH : ${ASCEND_HOME_PATH}")
message("== ASCEND_DRIVER_PATH : ${ASCEND_DRIVER_PATH}")