# shmem库编译-device能力
set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend910B3" CACHE STRING "system on chip type")

# 遗留问题: 此处一定要选Debug?
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()

file(GLOB KERNEL_FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/test_scalar_p/test_scalar_p_kernel.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test_team/test_team_kernel.cpp
)

set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}" CACHE STRING "ASCEND_CANN_PACKAGE_PATH")
set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
ascendc_library(test_scalar_npu SHARED ${KERNEL_FILES})
ascendc_include_directories(test_scalar_npu
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/host/
    ${PROJECT_SOURCE_DIR}/include/device/
    ${PROJECT_SOURCE_DIR}/include/host_device/
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include/host/
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include/aicore/
)

install(TARGETS test_scalar_npu
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

#put_get接口测试
add_executable(test_scalar_p
    ${CMAKE_CURRENT_SOURCE_DIR}/test_scalar_p/test_scalar_p_host.cpp
)

target_compile_options(test_scalar_p
    PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
)

target_include_directories(test_scalar_p
    PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include
)

target_link_directories(test_scalar_p
    PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/lib
)

target_link_libraries(test_scalar_p
    PRIVATE
    shmem
    test_scalar_npu
    mf_smem
    mf_hybrid_bmm
)

# team能力测试
add_executable(team_example
    ${CMAKE_CURRENT_SOURCE_DIR}/test_team/test_team_host.cpp
)

target_compile_options(team_example
    PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
)

target_include_directories(team_example
    PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include
)

target_link_directories(team_example
    PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid
)

target_link_libraries(team_example
    PRIVATE
    shmem
    test_scalar_npu
    mf_smem
    mf_hybrid_bmm
)

add_subdirectory(unittest)
