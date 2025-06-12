file(GLOB_RECURSE SHMEM_HOST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/init/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/mem/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/team/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/under_api/*.cpp"
)

if (BUILD_PYTHON STREQUAL "ON")
    add_subdirectory(python_wrapper)
endif ()

add_library(shmem_host SHARED)

target_sources(shmem_host PRIVATE ${SHMEM_HOST_SOURCES})

target_include_directories(shmem_host
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/init
    ${CMAKE_CURRENT_SOURCE_DIR}/mem
    ${CMAKE_CURRENT_SOURCE_DIR}/team
    ${CMAKE_CURRENT_SOURCE_DIR}/../device
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include/smem/host/
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/hccl/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/experiment
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/experiment/runtime/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/experiment/msprof/
)

target_link_directories(shmem_host
    PUBLIC
    ${ASCEND_DRIVER_PATH}/lib64/driver
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/lib
    ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64/
)

target_link_libraries(shmem_host
    PUBLIC
    ${MPI_CXX_LIBRARIES}
    ascendcl runtime profapi c_sec ascend_hal mf_smem mf_hybm_core
)

# 安装配置
install(TARGETS shmem_host
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

install (DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)