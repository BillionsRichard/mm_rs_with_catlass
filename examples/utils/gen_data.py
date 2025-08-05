import torch

from utils import CommType, DataType, tensor_to_file

def gen_random_data(size, dtype):
    if dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32:
        return torch.randn(size=size, dtype=dtype)
    elif dtype == torch.int8:
        return torch.randint(-16, 16, size=size, dtype=dtype)
    else:
        print(f"Invalid dtype: {dtype}.")
        exit(1)

def gen_golden_data():
    import argparse
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
    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k

    a_gm = gen_random_data([M, K], dtype=args.out_dtype.torch_type)
    b_gm = gen_random_data([K, N], dtype=args.out_dtype.torch_type)
    c_gm = torch.zeros((M, N), dtype=args.out_dtype.torch_type)

    l0c_dtype = torch.float32
    matrix_c = torch.matmul(a_gm.to(l0c_dtype), b_gm.to(l0c_dtype))

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

    if args.transA:
        a_gm = a_gm.transpose(0, 1).contiguous()
    if args.transB:
        b_gm = b_gm.transpose(0, 1).contiguous()

    tensor_to_file(a_gm, "./output/a_gm.bin")
    tensor_to_file(b_gm, "./output/b_gm.bin")
    tensor_to_file(c_gm, "./output/c_gm.bin")
    tensor_to_file(golden, "./output/golden.bin")

if __name__ == '__main__':
    gen_golden_data()
