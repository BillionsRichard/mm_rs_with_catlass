#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import numpy as np
import os

def gen_random_data(size, dtype):
    if dtype == np.float16 or dtype == np.float32:
        return np.random.uniform(size=size).astype(dtype)
    else:
        print(f"Invalid dtype: {dtype}.")
        exit(1)

def gen_golden_data():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        help='Directory to save the data files', 
                        default="./out")
    parser.add_argument('--out_data_type', type=int)
    parser.add_argument('--rank_size', type=int)
    parser.add_argument('--m', type=int)
    parser.add_argument('--n', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--transA', type=int)
    parser.add_argument('--transB', type=int)
    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k
    data_dir = os.path.abspath(args.data_dir)

    os.makedirs(data_dir, exist_ok=True)
    out_data_type = np.float32 if args.out_data_type == 0 else np.float16
    l0c_dtype = np.float32  # Use float32 for more precise matmul calculation

    golden = np.zeros((M, N), dtype=l0c_dtype)
    np.random.seed(42)

    for i in range(args.rank_size):
        # Using float16 for a and b as in the original script
        a_gm = gen_random_data((M, K), np.float16)
        b_gm = gen_random_data((K, N), np.float16)
        # print(f'rank_{i}, {a_gm=}, \n{b_gm=}')

        # Save per-rank data
        a_gm_path = os.path.join(data_dir, f"rank_{i}_a.bin")
        b_gm_path = os.path.join(data_dir, f"rank_{i}_b.bin")
        print(f'{a_gm_path=}')
        print(f'{b_gm_path=}')
        a_gm.tofile(a_gm_path)
        b_gm.tofile(b_gm_path)

        # Calculate matmul for this rank and add to golden
        matrix_c = np.matmul(a_gm.astype(l0c_dtype), b_gm.astype(l0c_dtype))
        golden += matrix_c

    # Convert to target data type and save golden data
    golden = golden.astype(out_data_type)
    golden_path = os.path.join(data_dir, "golden.bin")
    print(f'{golden_path=}')
    golden.tofile(golden_path)

if __name__ == '__main__':
    gen_golden_data()
