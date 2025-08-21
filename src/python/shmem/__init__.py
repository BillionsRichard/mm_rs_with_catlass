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
libs_path = os.path.join(os.getenv('SHMEM_HOME_PATH', current_dir), 'shmem/lib')
for lib in ["libshmem.so"]:
    ctypes.CDLL(os.path.join(libs_path, lib))

from ._pyshmem import shmem_init, shmem_finialize, shmem_malloc, shmem_free, \
    shmem_ptr, my_pe, pe_count, register_decrypt_handler, mte_set_ub_params, \
    team_split_strided, team_translate_pe, team_destroy, set_conf_store_tls

__all__ = [
    'shmem_init',
    'shmem_finialize',
    'shmem_malloc',
    'shmem_free',
    'shmem_ptr',
    'my_pe',
    'pe_count',
    'register_decrypt_handler',
    'set_conf_store_tls',
    'mte_set_ub_params',
    'team_split_strided',
    'team_translate_pe',
    'team_destroy'
]
