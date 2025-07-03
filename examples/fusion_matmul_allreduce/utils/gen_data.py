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
    parser.add_argument('out_data_type', type=int)
    parser.add_argument('rank_size', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('transA', type=int)
    parser.add_argument('transB', type=int)
    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k

    out_data_type = np.float32 if args.out_data_type == 0 else np.float16

    a_gm = gen_random_data((M, K), np.float16)
    b_gm = gen_random_data((K, N), np.float16)
    c_gm = np.zeros((M, N), dtype=np.float16)

    l0c_dtype = np.float32
    matrix_c = np.matmul(a_gm.astype(l0c_dtype), b_gm.astype(l0c_dtype)).astype(out_data_type)

    golden = np.zeros_like(matrix_c)
    for _ in range(args.rank_size):
        golden += matrix_c

    a_gm.tofile("./out/a_gm.bin")
    b_gm.tofile("./out/b_gm.bin")
    c_gm.tofile("./out/c_gm.bin")
    golden.tofile("./out/golden.bin")

if __name__ == '__main__':
    gen_golden_data()
