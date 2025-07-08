# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

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
endfunction()

foreach(EXAMPLE
    collective_allgather
    fusion_matmul_allreduce
)
    add_subdirectory(${EXAMPLE})
endforeach()