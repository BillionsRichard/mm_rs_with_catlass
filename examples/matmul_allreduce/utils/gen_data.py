import torch
import torch_npu
import numpy as np
import os

def gen_random_data(size, dtype):
    if dtype == torch.float16 or dtype == torch.float32:
        return torch.randn(size=size, dtype=dtype, device='cpu')
    elif dtype == torch.int8:
        return torch.randint(-16, 16, size=size, dtype=dtype, device='cpu')
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

    out_data_type = torch.float32 if args.out_data_type == 0 else torch.float16

    a_gm = gen_random_data([M, K], torch.float16)
    b_gm = gen_random_data([K, N], torch.float16)
    c_gm = torch.zeros(size=[M, N], dtype=torch.float16, device='cpu')

    l0c_dtype = torch.float32
    matrix_c = torch.matmul(a_gm.to(l0c_dtype), b_gm.to(l0c_dtype)).to(out_data_type)

    golden = torch.zeros_like(matrix_c)
    for _ in range(args.rank_size):
        golden += matrix_c

    if args.transA:
        a_gm = a_gm.transpose(0, 1).contiguous()
    if args.transB:
        b_gm = b_gm.transpose(0, 1).contiguous()

    a_gm.numpy().tofile("./out/a_gm.bin")
    b_gm.numpy().tofile("./out/b_gm.bin")
    c_gm.numpy().tofile("./out/c_gm.bin")
    golden.numpy().tofile("./out/golden.bin")

if __name__ == '__main__':
    gen_golden_data()
