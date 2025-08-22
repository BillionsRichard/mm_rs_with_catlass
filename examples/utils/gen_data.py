import torch
import os
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
                        choices=[CommType.MATMUL_ALLREDUCE,
                                 CommType.ALLGATHER_MATMUL,
                                 CommType.MATMUL_REDUCE_SCATTER,
                                 CommType.ALLGATHER_MATMUL_WITH_GATHER_RESULT])
    parser.add_argument('out_dtype', type=DataType.from_str, choices=[DataType.FLOAT16, DataType.BF16])
    parser.add_argument('rank_size', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('transA', type=int)
    parser.add_argument('transB', type=int)
    parser.add_argument('data_dir', type=str,
                        help='Directory to save the data files',
                        default="./out")
    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k
    data_dir = os.path.abspath(args.data_dir)

    os.makedirs(data_dir, exist_ok=True)
    b_all_rank = gen_random_data([K, N], dtype=args.out_dtype.torch_type)

    l0c_dtype = torch.float32
    matrix_a_list = []
    matrix_c_list = []
    for i in range(args.rank_size):
        a_gm = gen_random_data([M, K], dtype=args.out_dtype.torch_type)
        matrix_a_list.append(a_gm)
        b_gm = b_all_rank
        matrix_c = torch.matmul(a_gm.to(l0c_dtype), b_gm.to(l0c_dtype))
        matrix_c_list.append(matrix_c)
        if args.transA:
            a_gm = a_gm.transpose(0, 1).contiguous()
        if args.transB:
            b_gm = b_gm.transpose(0, 1).contiguous()

        a_gm_path = os.path.join(data_dir, f"rank_{i}_a.bin")
        b_gm_path = os.path.join(data_dir, f"rank_{i}_b.bin")
        tensor_to_file(a_gm, a_gm_path)
        tensor_to_file(b_gm, b_gm_path)

    golden = None
    if args.comm_type in [CommType.ALLGATHER_MATMUL, CommType.ALLGATHER_MATMUL_WITH_GATHER_RESULT]:
        golden = torch.cat(matrix_c_list, dim=0)
    else:
        golden = torch.zeros_like(matrix_c_list[0])
        for i in range(args.rank_size):
            golden += matrix_c_list[i]

    tensor_to_file(golden, os.path.join(data_dir, "golden.bin"))

    if args.comm_type == CommType.ALLGATHER_MATMUL_WITH_GATHER_RESULT:
        tensor_to_file(torch.cat(matrix_a_list, dim=0).to(torch.float32), os.path.join(data_dir, "gather_a.bin"))

if __name__ == '__main__':
    gen_golden_data()