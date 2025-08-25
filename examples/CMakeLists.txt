# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

function(shmem_add_fusion_example NAME)
    add_executable(${NAME} ${ARGN})

    target_compile_options(${NAME} PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220)
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/install/memfabric_hybrid/include/smem/host/
        ${PROJECT_SOURCE_DIR}/install/memfabric_hybrid/include/smem/device/
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/include
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
        ${PROJECT_SOURCE_DIR}/examples/${NAME}
        ${PROJECT_SOURCE_DIR}/examples/templates/include
        ${PROJECT_SOURCE_DIR}/examples/utils
    )
    target_link_options(${NAME} PRIVATE --cce-fatobj-link)
    target_link_libraries(${NAME} PRIVATE shmem ${PROJECT_SOURCE_DIR}/install/memfabric_hybrid/lib/libmf_smem.so)
endfunction()

function(shmem_add_collective_example NAME)
    add_library(${NAME}_kernel SHARED ${NAME}_kernel.cpp)
    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220-vec)
    target_include_directories(${NAME}_kernel PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/install/memfabric_hybrid/include/smem/device/
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/include
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
        ${PROJECT_SOURCE_DIR}/examples/${NAME}
    )
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)

    add_executable(${NAME} main.cpp)
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/install/memfabric_hybrid/include/smem/host/
        ${PROJECT_SOURCE_DIR}/3rdparty/catlass/examples/common
        ${PROJECT_SOURCE_DIR}/examples/${NAME}
        ${PROJECT_SOURCE_DIR}/examples/templates/include
        ${PROJECT_SOURCE_DIR}/examples/utils
    )
    target_link_libraries(${NAME} PRIVATE shmem ${NAME}_kernel ${PROJECT_SOURCE_DIR}/install/memfabric_hybrid/lib/libmf_smem.so)
endfunction()

foreach(EXAMPLE
    allgather
    allgather_matmul
    allgather_matmul_dequant
    allgather_matmul_with_gather_result
    matmul_allreduce
    matmul_reduce_scatter
    # matmul_reduce_scatter_quant_perchn
    matmul_reduce_scatter_dequant
    dynamic_tiling
)
    add_subdirectory(${EXAMPLE})
endforeach()