# kernel
file(GLOB KERNEL_FILES test_barrier_kernel.cpp)
# include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
ascendc_library(test_barrier_kernel SHARED ${KERNEL_FILES})

ascendc_include_directories(test_barrier_kernel
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/include/host/
    ${PROJECT_SOURCE_DIR}/include/device/
    ${PROJECT_SOURCE_DIR}/include/host_device/
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include/smem/host/
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include/smem/device/
)

install(TARGETS test_barrier_kernel
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

# executable
add_executable(test_barrier test_barrier_host.cpp)
target_compile_options(test_barrier
    PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
)

target_include_directories(test_barrier
    PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include
)

target_link_directories(test_barrier
    PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/lib
)

target_link_libraries(test_barrier
    PRIVATE
    shmem_host
    shmem_device
    test_barrier_kernel
    mf_smem
    mf_hybm_core
)
