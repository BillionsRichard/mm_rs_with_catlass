#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import sys
import numpy as np

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

def is_close(actual: np.ndarray, expected: np.ndarray, rtol: float = 2 ** -8):
    if actual.dtype == np.float16 or actual.dtype == np.float32:
        return np.abs(actual - expected) <= rtol * np.maximum(1, np.abs(expected))
    elif actual.dtype == np.int32:
        return actual == expected

def verify_result():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str)
    parser.add_argument('golden', type=str)
    parser.add_argument('out_data_type', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    args = parser.parse_args()
    m, n = args.m, args.n

    out_data_type = np.float32 if args.out_data_type == 0 else np.float16

    output = np.fromfile(args.output, dtype=out_data_type)
    golden = np.fromfile(args.golden, dtype=out_data_type)
    output = output.reshape(-1)
    golden = golden.reshape(-1)

    rtol = 2 ** -8 if args.k < 2048 else 2 ** -7
    different_element_result = np.logical_not(is_close(output, golden, rtol))

    different_element_indexes = np.where(different_element_result)[0]

    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print(f"data index: {real_index:6d}, expected: {golden_data:-.9f}, actual: {output_data:-.9f}, "
            f"rdiff: {abs(output_data - golden_data) / golden_data:-.6f}")
        if index == 10:
            break
    error_num = different_element_indexes.size
    print(f"error num: {error_num}")
    return error_num == 0

if __name__ == '__main__':
    try:
        res = verify_result()
        if not res:
            print(f"{RED}ERROR{RESET}")
        else:
            print(f"{GREEN}PASS{RESET}")
    except Exception as e:
        print(e)
        sys.exit(1)
