#shmem库编译-host能力
set(SHMEM_HOST_SOURCES
    domain/team.cpp
    init/init.cpp
    mem/mem.cpp
)

add_library(
    shmem SHARED
)

target_sources(shmem PRIVATE ${SHMEM_HOST_SOURCES})

target_include_directories(shmem
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/host/>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/host_device/>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/host>
    $<INSTALL_INTERFACE:include/host_device>
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include/host/
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/hccl/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/experiment
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/experiment/runtime/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/experiment/msprof/
)

target_link_directories(shmem
    PUBLIC
    ${ASCEND_DRIVER_PATH}/lib64/driver
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/lib
    ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64/
)

target_link_libraries(shmem
    PUBLIC
    ${MPI_CXX_LIBRARIES}
    ascendcl runtime profapi c_sec ascend_hal mf_smem mf_hybrid_bmm
)

# 安装配置
install(TARGETS shmem
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

install (FILES
    ${PROJECT_SOURCE_DIR}/include/host/init.h
    ${PROJECT_SOURCE_DIR}/include/host/team.h
    DESTINATION include
)