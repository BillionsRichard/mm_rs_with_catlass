# Compile kernel
set(CCEC_COMPILER_OPTIONS
    -O2 -std=c++17 -xcce --cce-aicore-arch=dav-c220
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
)
set(LIB_OPTIONS
    --shared -fPIC
)
set(INCLUDE_DIR
    -I${CMAKE_SOURCE_DIR}/include
    -I${CMAKE_SOURCE_DIR}/examples
    -I${CMAKE_SOURCE_DIR}/3rdparty/catlass/include
    -I${ASCEND_HOME_PATH}/include
    -I${ASCEND_HOME_PATH}/compiler/ascendc/include/basic_api
    -I${ASCEND_HOME_PATH}/compiler/ascendc/include/basic_api/impl
    -I${ASCEND_HOME_PATH}/compiler/ascendc/include/basic_api/interface
    -I${ASCEND_HOME_PATH}/compiler/ascendc/include/highlevel_api
    -I${ASCEND_HOME_PATH}/include/experiment/runtime
    -I${ASCEND_HOME_PATH}/include/experiment/msprof
    -I${SHMEM_HOME_PATH}/shmem/include
    -I${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host
    -I${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
    -I${CMAKE_CURRENT_SOURCE_DIR}
    -I${CMAKE_CURRENT_SOURCE_DIR}/include
)
set(LINK_LIB
    -L${ASCEND_HOME_PATH}/lib64
    -L${SHMEM_HOME_PATH}/memfabric_hybrid/lib
    -L${SHMEM_HOME_PATH}/shmem/lib
    -L${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    -Wl,-rpath,${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    -lruntime -lstdc++ -lascendcl
    -lmf_smem -lmf_hybm_core -lshmem
)
file(GLOB_RECURSE CATLASS_DEPEND_FILES ${CMAKE_SOURCE_DIR}/include/**.hpp)

add_custom_target(lib_impl)
function(add_impl_share_lib NAME)
    add_custom_command(
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib${NAME}.so
        COMMAND ${CCEC} ${CCEC_COMPILER_OPTIONS} ${LIB_OPTIONS} ${INCLUDE_DIR}
            ${NAME}.cpp -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib${NAME}.so
        DEPENDS ${NAME}.cpp ${CMAKE_CURRENT_SOURCE_DIR}/kernel/*.h ${CATLASS_DEPEND_FILES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Compiling shared library: lib${NAME}.so"
    )
    add_custom_target(lib${NAME}.so ALL DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib${NAME}.so)
    add_dependencies(lib_impl lib${NAME}.so)
endfunction()

add_subdirectory(impl)

include_directories(${ASCEND_HOME_PATH}/include/experiment/runtime 
    ${ASCEND_HOME_PATH}/include/experiment/msprof
    ${ASCEND_HOME_PATH}/include 
    ${CMAKE_SOURCE_DIR}/include 
    ${CMAKE_SOURCE_DIR}/examples
    ${SHMEM_HOME_PATH}/shmem/include
    ${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${CMAKE_RUNTIME_OUTPUT_DIRECTORY} ${ASCEND_HOME_PATH}/lib64 
    ${SHMEM_HOME_PATH}/memfabric_hybrid/lib ${SHMEM_HOME_PATH}/shmem/lib)
add_executable(dynamic_tiling dynamic_tiling.cpp)
target_link_libraries(dynamic_tiling runtime ascendcl mf_smem mf_hybm_core shmem ${SHARE_LIB_LINK})
set_target_properties(dynamic_tiling PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

add_dependencies(dynamic_tiling lib_impl)