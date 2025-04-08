# shmem库编译-device能力
set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend910B3" CACHE STRING "system on chip type")

# 遗留问题: 此处一定要选Debug?
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()

file(GLOB KERNEL_FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/comm/scalar_p.cpp
)

set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}" CACHE STRING "ASCEND_CANN_PACKAGE_PATH")
set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)

# 底层接口增加 ———— 显式修改CANN包内容，是否妥当？
set(ascendc_include_dir ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api/interface/)
exec_program(cp ${CMAKE_CURRENT_SOURCE_DIR} ARGS -rvf ${CMAKE_CURRENT_SOURCE_DIR}/memfabric_hybrid/src/smem/include/aicore/* ${ascendc_include_dir})

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
ascendc_library(ascendc_kernels_npu SHARED ${KERNEL_FILES})

install(TARGETS ascendc_kernels_npu
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)