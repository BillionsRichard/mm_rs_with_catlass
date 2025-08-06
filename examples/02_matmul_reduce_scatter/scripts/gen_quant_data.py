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
    parser.add_argument('out_dtype', type=DataType.from_str, choices=[DataType.FLOAT16, DataType.BF16]) # [fp16:1, bf16:27]
    parser.add_argument('rank_size', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('transA', type=int)
    parser.add_argument('transB', type=int)
    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k

    # Generate quantized inputs
    x1_gm = gen_random_data([M, K], dtype=torch.int8)
    x2_gm = gen_random_data([K, N], dtype=torch.int8)
    # scale_x1_gm = torch.randn(M, dtype=torch.float32) * 0.01
    # scale_x2_gm = torch.randn(N, dtype=torch.float32) * 0.01    
    scale_x1_gm = torch.ones(size=(M,), dtype=torch.float32) * 0.01
    scale_x2_gm = torch.ones(size=(N,), dtype=torch.float32) * 0.01
    # bias_gm = torch.randint(-16, 16, size=(N,), dtype=torch.int32)
    bias_gm = torch.zeros(size=(N,), dtype=torch.int32)
    c_gm = torch.zeros((M // args.rank_size, N), dtype=args.out_dtype.torch_type)

    # Calculate golden result
    l0c_dtype = torch.float32
    accumulator = torch.matmul(x1_gm.to(l0c_dtype), x2_gm.to(l0c_dtype))
    
    dequantized = accumulator + bias_gm.to(l0c_dtype)
    
    # Apply scales
    # scale_x1 is per-token (per-row of A), scale_x2 is per-channel (per-column of B)
    result_fp32 = dequantized * scale_x1_gm.unsqueeze(1).to(l0c_dtype) * scale_x2_gm.unsqueeze(0).to(l0c_dtype)

    # The reduce-scatter operation sums the results from all ranks
    golden = torch.zeros_like(result_fp32)
    for _ in range(args.rank_size):
        golden += result_fp32
    
    golden = golden.to(args.out_dtype.torch_type)

    if args.transA:
        x1_gm = x1_gm.transpose(0, 1).contiguous()
    if args.transB:
        x2_gm = x2_gm.transpose(0, 1).contiguous()

    tensor_to_file(x1_gm, "./output/x1_gm.bin") # int8
    tensor_to_file(x2_gm, "./output/x2_gm.bin") # int 8
    tensor_to_file(scale_x1_gm, "./output/scale_x1_gm.bin") # fp32
    tensor_to_file(scale_x2_gm, "./output/scale_x2_gm.bin") # fp32
    tensor_to_file(bias_gm, "./output/bias_gm.bin") # int32
    tensor_to_file(c_gm, "./output/c_gm.bin") # This is a placeholder for the output buffer
    tensor_to_file(golden, "./output/golden.bin") # fp16

if __name__ == '__main__':
    gen_golden_data()
