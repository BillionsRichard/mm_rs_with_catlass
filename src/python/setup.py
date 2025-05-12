#!/usr/bin/env python
# coding=utf-8
# Copyright: (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

"""python api for aclshmem."""

import os

import setuptools
from setuptools import find_namespace_packages
from setuptools.dist import Distribution

# 消除whl压缩包的时间戳差异
os.environ['SOURCE_DATE_EPOCH'] = '0'

current_version = os.getenv('VERSION', '1.0.0')


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(self):
        return True


setuptools.setup(
    name="aclshmem",
    version=current_version,
    author="",
    author_email="",
    description="python api for aclshmem",
    packages=find_namespace_packages(exclude=("tests*",)),
    url="https://open.codehub.huawei.com/innersource/OpenComputingKit_G/MatrixMemory/shmem",
    license="Apache License Version 2.0",
    install_requires=["torch-npu"],
    python_requires=">=3.7",
    package_data={"aclshmem": ["_pyaclshmem.cpython*.so", "lib/**", "VERSION"]},
    distclass=BinaryDistribution
)
