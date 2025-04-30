include_directories(
    ${PROJECT_SOURCE_DIR}/install/shmem/include/
    ${PROJECT_SOURCE_DIR}/src/include/
    ${PROJECT_SOURCE_DIR}/install/memfabric_hybrid/include/host/
)

file(GLOB_RECURSE KERNEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*_kernel.cpp")

ascendc_library(unittest_kernel SHARED ${KERNEL_FILES})
ascendc_include_directories(unittest_kernel
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/install/memfabric_hybrid/include/smem/device/
)

ascendc_compile_definitions(unittest_kernel PRIVATE
    -DASCENDC_DUMP=1
)

install(TARGETS unittest_kernel
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

add_subdirectory(unittest)
