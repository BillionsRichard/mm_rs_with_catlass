#shmem库编译-host能力
set(SHMEM_HOST_SOURCES
    mem/shmem_heap.cpp
    domain/team.cpp
    init/init.cpp
)

add_library(
    shmem SHARED
)

target_sources(shmem PRIVATE ${SHMEM_HOST_SOURCES})

target_include_directories(shmem
    PUBLIC
    ${MPI_CXX_INCLUDE_PATH}
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
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
    ${ASCEND_HOME_PATH}/${ARCH}-linux/lib64/
)


target_link_libraries(shmem
    PUBLIC
    ${MPI_CXX_LIBRARIES}
    ascendcl runtime profapi c_sec ascend_hal
)

# 安装配置
install(TARGETS shmem
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

install (FILES
    ${PROJECT_SOURCE_DIR}/include/init.h
    ${PROJECT_SOURCE_DIR}/include/shmem_heap.h
    ${PROJECT_SOURCE_DIR}/include/team.h
    DESTINATION include
)