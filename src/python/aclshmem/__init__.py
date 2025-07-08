#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import os
import sys
import ctypes
import torch_npu


current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
sys.path.append(current_dir)
libs_path = os.path.join(current_dir, 'lib')
for lib in ["libaclshmem_host.so"]:
    ctypes.CDLL(os.path.join(libs_path, lib))

from ._pyaclshmem import set_logger_level, aclshmem_init, aclshmem_finialize, aclshmem_malloc, aclshmem_free, \
    aclshmem_ptr, my_pe, pe_count, mte_set_ub_params, team_split_strided, team_translate_pe, team_destroy, \
    barrier_all, barrier_team

__all__ = [
    'set_logger_level',
    'aclshmem_init',
    'aclshmem_finialize',
    'aclshmem_malloc',
    'aclshmem_free',
    'aclshmem_ptr',
    'my_pe',
    'pe_count',
    'mte_set_ub_params',
    'team_split_strided',
    'team_translate_pe',
    'team_destroy',
    'barrier_all',
    'barrier_team'
]
