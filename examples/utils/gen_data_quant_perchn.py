import os.path
import shutil

import torch
import argparse
from utils import CommType, DataType, tensor_to_file
import numpy as np


def gen_random_data(size, dtype):
    if dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32:
        return torch.randn(size=size, dtype=dtype)
    elif dtype == torch.int8 or dtype == torch.int32:
        return torch.randint(-16, 16, size=size, dtype=dtype)
    else:
        print(f"Invalid dtype: {dtype}.")
        exit(1)


def gen_golden_data(args):
    M, N, K = args.m, args.n, args.k
    args.in_dtype = args.in_dtype if (args.bias or args.scale) else args.out_dtype

    a_gm = gen_random_data([M, K], dtype=args.in_dtype.torch_type)
    b_gm = gen_random_data([K, N], dtype=args.in_dtype.torch_type)
    c_gm = torch.zeros((M, N), dtype=args.out_dtype.torch_type)
    bias_gm = gen_random_data([N], dtype=torch.int32)
    scale_gm = torch.rand((N), dtype=torch.float)

    matrix_c = torch.matmul(a_gm.to(torch.int32), b_gm.to(torch.int32))
    if args.bias:
        matrix_c = torch.add(matrix_c.to(bias_gm.dtype), bias_gm)
    if args.scale:
        matrix_c = torch.mul(matrix_c.to(scale_gm.dtype), scale_gm)

    golden = None
    if args.comm_type == CommType.ALLGATHER_MATMUL:
        matrix_c_list = []
        for _ in range(args.rank_size):
            matrix_c_list.append(matrix_c)
        golden = torch.cat(matrix_c_list, dim=0)
    else:
        golden = torch.zeros_like(matrix_c)
        for _ in range(args.rank_size):
            golden += matrix_c
    golden = golden.to(torch.float32)

    if args.transA:
        a_gm = a_gm.transpose(0, 1).contiguous()
    if args.transB:
        b_gm = b_gm.transpose(0, 1).contiguous()

    tensor_to_file(a_gm, "./output/a_gm.bin")
    tensor_to_file(b_gm, "./output/b_gm.bin")
    tensor_to_file(c_gm, "./output/c_gm.bin")
    tensor_to_file(matrix_c.to(torch.float16), "./output/matrix_c.bin")
    tensor_to_file(golden, "./output/golden.bin")
    if args.bias:
        tensor_to_file(bias_gm, "./output/bias_gm.bin")
    if args.scale:
        std_scale = scale_gm.numpy().view(np.uint32).astype(np.uint64)
        std_scale.tofile("./output/scale_gm.bin")
        # print(std_scale)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('comm_type', type=CommType.from_str,
                        choices=[CommType.MATMUL_ALLREDUCE, CommType.ALLGATHER_MATMUL, CommType.MATMUL_REDUCE_SCATTER])
    parser.add_argument('out_dtype', type=DataType.from_str, choices=[DataType.FLOAT16, DataType.BF16])
    parser.add_argument('rank_size', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('transA', type=int)
    parser.add_argument('transB', type=int)
    parser.add_argument('bias', type=int, default=0)
    parser.add_argument('scale', type=int, default=0)
    parser.add_argument('in_dtype', type=DataType.from_str,
                        choices=[DataType.INT8, DataType.FLOAT16, DataType.BF16, DataType.FLOAT], default=DataType.FLOAT)
    args = parser.parse_args()
    args.in_dtype = args.in_dtype if (args.bias or args.scale) else args.out_dtype

    M, N, K = args.m, args.n, args.k

    if os.path.isfile("./output/golden.bin"):
        shutil.rmtree("./output")
        os.mkdir("./output")
    gen_golden_data(args)

    # if not os.path.isfile("./output/golden.bin"):
    #     gen_golden_data(args)
    # else:
    #     print('Using existing dump tensor!')
    #     a_gm = np.fromfile("./output/a_gm.bin", dtype=args.in_dtype.numpy_type).reshape(M, K)
    #     b_gm = np.fromfile("./output/b_gm.bin", dtype=args.in_dtype.numpy_type).reshape(K, N)
    #     c_gm = np.fromfile("./output/c_gm.bin", dtype=args.out_dtype.numpy_type).reshape(M, N)
    #     matrix_c = np.fromfile("./output/matrix_c.bin", dtype=args.out_dtype.numpy_type).reshape(M, N)
    #     bias_gm = np.fromfile("./output/bias_gm.bin", dtype=np.int32) if args.bias else None
    #     scale_gm = np.fromfile("./output/scale_gm.bin", dtype=np.uint64) if args.scale else None
    #     golden = np.fromfile("./output/golden.bin", dtype=args.out_dtype.numpy_type)
