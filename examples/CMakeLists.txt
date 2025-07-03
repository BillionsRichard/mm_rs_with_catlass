
function(shmem_add_example NAME)
    add_executable(${NAME} ${ARGN})

    if(${NAME} MATCHES "collective")
        # TODO: dav-c220-vec
        set(ARCH "dav-c220")
    else()
        set(ARCH "dav-c220")
    endif()

    target_compile_options(${NAME} PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=${ARCH})
    target_include_directories(${NAME} PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/output/smem/include
        ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/output/smem/include/smem/host/
        ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/output/smem/include/smem/device/
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/include
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
        ${PROJECT_SOURCE_DIR}/examples/${NAME}
    )
    target_link_options(${NAME} PRIVATE --cce-fatobj-link)
    target_link_libraries(${NAME} PRIVATE shmem)
    target_link_libraries(${NAME} PRIVATE runtime stdc++ ascendcl m tiling_api platform c_sec dl nnopbase)
    target_link_libraries(${NAME} PRIVATE ascendc_runtime error_manager profapi mmpa ascendalog)
endfunction()

foreach(EXAMPLE
    collective_allgather
    fusion_matmul_allreduce
)
    add_subdirectory(${EXAMPLE})
endforeach()