if (BUILD_PYTHON STREQUAL "ON")
    add_subdirectory(host/python_wrapper)
endif ()

file(GLOB_RECURSE SHMEM_KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/device/*.cpp)
add_library(shmem_device OBJECT ${SHMEM_KERNEL_FILES})
target_compile_options(shmem_device PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220)
target_include_directories(shmem_device
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/output/smem/include
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/output/smem/include/smem/device/
)

file(GLOB_RECURSE SHMEM_HOST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/host/*.cpp)
list(FILTER SHMEM_HOST_FILES EXCLUDE REGEX "python_wrapper")
add_library(shmem_host OBJECT ${SHMEM_HOST_FILES})
target_compile_options(shmem_host PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(shmem_host
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/output/smem/include
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/output/smem/include/smem/host/
    ${PROJECT_SOURCE_DIR}/src/host
    ${PROJECT_SOURCE_DIR}/src/device
)

add_library(shmem SHARED $<TARGET_OBJECTS:shmem_device> $<TARGET_OBJECTS:shmem_host>)
target_link_options(shmem PRIVATE --cce-fatobj-link)

# 安装配置
install(TARGETS shmem
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

install (DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)