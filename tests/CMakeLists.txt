# shmem库编译-device能力
set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend910B3" CACHE STRING "system on chip type")

# 遗留问题: 此处一定要选Debug?
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()

file(GLOB_RECURSE KERNEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*_kernel.cpp")

set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}" CACHE STRING "ASCEND_CANN_PACKAGE_PATH")
set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
ascendc_library(test_scalar_npu SHARED ${KERNEL_FILES})
ascendc_include_directories(test_scalar_npu
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/host/
    ${PROJECT_SOURCE_DIR}/include/device/
    ${PROJECT_SOURCE_DIR}/include/host_device/
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include/smem/host/
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include/smem/device/
)

ascendc_compile_definitions(test_scalar_npu PRIVATE
    -DASCENDC_DUMP=1
)

install(TARGETS test_scalar_npu
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

add_subdirectory(unittest)
