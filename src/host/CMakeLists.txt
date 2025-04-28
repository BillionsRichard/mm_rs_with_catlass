file(GLOB SHMEM_HOST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(shmem_host SHARED)

target_sources(shmem_host PRIVATE ${SHMEM_HOST_SOURCES})

target_include_directories(shmem_host
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/src/include/
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

install (FILES
    ${PROJECT_SOURCE_DIR}/include/host/shmem_api.h
    ${PROJECT_SOURCE_DIR}/include/host/team.h
    DESTINATION include
)