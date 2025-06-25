file(GLOB_RECURSE TEST_KERNEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*_kernel.cpp")
file(GLOB_RECURSE TEST_HOST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp")

add_library(shmem_unittest_kernel OBJECT ${TEST_KERNEL_FILES})
target_compile_options(shmem_unittest_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS})
target_include_directories(shmem_unittest_kernel PRIVATE
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/output/smem/include
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/output/smem/include/smem/device
)

add_library(shmem_unittest_host OBJECT ${TEST_HOST_FILES})
target_compile_options(shmem_unittest_host PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(shmem_unittest_host PRIVATE
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/output/smem/include
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/output/smem/include/smem/host
    ${PROJECT_SOURCE_DIR}/src/host
    ${PROJECT_SOURCE_DIR}/src/device
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include
)

add_executable(shmem_unittest $<TARGET_OBJECTS:shmem_unittest_kernel> $<TARGET_OBJECTS:shmem_unittest_host>)
target_link_options(shmem_unittest PRIVATE --cce-fatobj-link)
target_link_directories(shmem_unittest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib)
target_link_libraries(shmem_unittest PRIVATE shmem gtest)
