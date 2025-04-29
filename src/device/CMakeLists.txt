file(GLOB KERNEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

ascendc_library(shmem_device SHARED ${KERNEL_FILES})

ascendc_include_directories(shmem_device
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/src/include/
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include/smem/host/
    ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include/smem/device/
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${ASCEND_HOME_PATH}/${ARCH}-linux/include/
)

install(TARGETS shmem_device
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)