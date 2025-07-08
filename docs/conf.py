#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import subprocess
import warnings

branch = subprocess.check_output(["/bin/bash", "-c", "git symbolic-ref -q --short HEAD || git describe --tags --exact-match 2> /dev/null || git rev-parse HEAD"]).strip().decode()
project = "Shmem Guidebook"
author = "xxx"
copyright = "2025"
release = "1.0.0"
html_show_sphinx = False

extensions = [
    'myst_parser',
    'breathe',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
]

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
}

breathe_projects = {"SHMEM_CPP_API": f"./{branch}/xml"}
breathe_default_project = "SHMEM_CPP_API"