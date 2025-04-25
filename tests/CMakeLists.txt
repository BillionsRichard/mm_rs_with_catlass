include_directories(
    ${PROJECT_SOURCE_DIR}/include/host/
    ${PROJECT_SOURCE_DIR}/include/device/
    ${PROJECT_SOURCE_DIR}/include/host_device/
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include/host/
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include/aicore/
)

file(GLOB_RECURSE KERNEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*_kernel.cpp")

ascendc_library(test_scalar_npu SHARED ${KERNEL_FILES})
ascendc_include_directories(test_scalar_npu
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/
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
add_subdirectory(test_barrier)
