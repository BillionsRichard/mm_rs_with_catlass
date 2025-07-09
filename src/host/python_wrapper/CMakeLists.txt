# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

project(pyshmem LANGUAGES CXX)

add_compile_options(-fabi-version=0)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

find_program(PYTHON_EXECUTABLE NAMES python3 python)

message(STATUS "PYTHON_EXECUTABLE:${PYTHON_EXECUTABLE}")

execute_process(
        COMMAND ${PYTHON_EXECUTABLE} "-c"
        "from __future__ import print_function; import os; import pybind11;
print(os.path.dirname(pybind11.__file__),end='');"
        RESULT_VARIABLE _PYTHON_SUCCESS
        OUTPUT_VARIABLE PYBIND11_DIR)

if(NOT _PYTHON_SUCCESS MATCHES 0)
    message(STATUS "PYBIND11_DIR: ${PYBIND11_DIR}")
    message(FATAL_ERROR "Pybind11 config Error.")
endif()

message(STATUS "PYBIND11_DIR = ${PYBIND11_DIR}")
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_DIR})
find_package(pybind11 REQUIRED)

# find torch
execute_process(
        COMMAND ${PYTHON_EXECUTABLE} "-c"
        "from __future__ import print_function; import os; import torch;
print(os.path.dirname(torch.__file__),end='');"
        RESULT_VARIABLE _TORCH_SUCCESS
        OUTPUT_VARIABLE TORCH_DIR)

if(NOT _TORCH_SUCCESS MATCHES 0)
    message(STATUS "PyTorch:${TORCH_DIR}")
    message(FATAL_ERROR "Torch config Error.")
endif()
message(STATUS "TORCH_DIR = ${TORCH_DIR}")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})

# find torch_npu
set(ENV{LD_LIBRARY_PATH} "${ASCEND_DRIVER_PATH}/lib64/driver:${ASCEND_HOME_PATH}/aarch64-linux/lib64")
execute_process(
        COMMAND ${PYTHON_EXECUTABLE} "-c"
        "from __future__ import print_function; import os; import torch_npu;
print(os.path.dirname(torch_npu.__file__),end='');"
        RESULT_VARIABLE _TORCH_NPU_SUCCESS
        OUTPUT_VARIABLE TORCH_NPU_DIR
)

message(STATUS "TORCH_NPU_DIR = ${TORCH_NPU_DIR}")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_NPU_DIR})

pybind11_add_module(_pyaclshmem pyshmem.cpp)
set_target_properties(_pyaclshmem PROPERTIES CXX_STANDARD 17)
target_include_directories(_pyaclshmem PRIVATE
        ${TORCH_DIR}/include
        ${TORCH_DIR}/include/torch/csrc/api/include
        ${TORCH_NPU_DIR}/include
        ${TORCH_NPU_DIR}/include/third_party/hccl/inc
        ${TORCH_NPU_DIR}/include/third_party/acl/inc
        ${TORCH_NPU_DIR}/include/torch_npu/csrc/core/npu
        ${SHMEM_SOURCE_ROOT}/include
)
target_link_directories(_pyaclshmem PRIVATE
        ${TORCH_DIR}/lib
        ${TORCH_NPU_DIR}/lib
)
target_link_libraries(_pyaclshmem PRIVATE
        shmem_host
        torch_npu
        torch_cpu
        c10
)

install(TARGETS _pyaclshmem
        LIBRARY DESTINATION ${TARGET_INSTALL_DIR}/lib
        PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
)
