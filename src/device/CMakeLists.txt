file(GLOB KERNEL_FILES sync/shmem_barrier.cpp)

ascendc_library(shmem_device STATIC ${KERNEL_FILES})

ascendc_include_directories(shmem_device
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/include/host/
    ${PROJECT_SOURCE_DIR}/include/device/
    ${PROJECT_SOURCE_DIR}/include/host_device/
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include/host/
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include/aicore/
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
)

install(TARGETS shmem_device
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)