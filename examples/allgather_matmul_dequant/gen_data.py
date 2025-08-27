import torch
import os

from utils import tensor_to_file

BIAS_LOW = -65536
BIAS_HIGH = 65536
WORKSPACE = os.getcwd()

os.environ["WORKSPACE"] = WORKSPACE

def gen_random_data(size, dtype, debug=False):
    if dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32:
        return torch.empty(size=size, dtype=dtype).uniform_(-2,2)
    elif dtype == torch.int8:
        if not debug:
            return torch.randint(-16, 16, size=size, dtype=dtype)
        else:
            return torch.ones(size=size, dtype=dtype)
    else:
        print(f"Invalid dtype: {dtype}.")
        exit(1)
        
def dequantize(golden, scale_gm, perTokenScale_gm, pertensor_scale=False):
    if not pertensor_scale:
        golden = golden.to(torch.float32) * (perTokenScale_gm * scale_gm)
    else:
        scale_gm *= perTokenScale_gm[0]
        golden = golden.to(torch.float32) * scale_gm

    return golden

           
def gen_golden_data():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('rank_size', type=int)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    args = parser.parse_args()
    rankSize = args.rank_size
    M, N, K = args.m, args.n, args.k

    debug = int(os.getenv("debug", 0)) == 1
    
    if not debug:
        per_channel_scale_gm = torch.empty(size=[1,N], dtype=torch.float32).uniform_(0.004,0.005)
        per_tensor_scale = torch.rand(1).item()
    else:
        per_tensor_scale = 0.1
        per_channel_scale_gm = torch.full(size=[1, N], fill_value=0.1, dtype=torch.float32)

    # use all same value as perTensorScale.
    per_token_scale_gm = torch.full(size=[M,1], fill_value=per_tensor_scale, dtype=torch.float32)
    fused_scale_gm = per_channel_scale_gm * per_tensor_scale
    print(f'{per_tensor_scale=}')
    print(f'{per_channel_scale_gm=}')
    print(f'{fused_scale_gm=}')
    d_gm = torch.zeros((M * rankSize, N), dtype= torch.float32)
    
    if debug:
        # print(f'[warning]: in debug mode, use {torch.arange(1, N+1, dtype=torch.int32)} as bias input.')
        # bias_gm = torch.arange(1, N+1, dtype=torch.int32)
        bias_gm = torch.zeros(size=(N,), dtype=torch.int32)
    else:
        bias_gm = torch.randint(low=BIAS_LOW, high=BIAS_HIGH+1, size=(N,), dtype=torch.int32)
    
    tensor_to_file(bias_gm, "./output/bias_gm.bin")

    A = []
    for i in range(args.rank_size):
        # Generate rank-specific inputs x1 and x2,
        a_gm_rank = gen_random_data([M, K], dtype=torch.int8, debug=debug)
        tensor_to_file(a_gm_rank, f"./output/a_gm_rank{i}.bin")
        print(f"Generated data a for rank {i}:")
        print(f"  a_gm_rank{i}.bin: shape={a_gm_rank.shape}")
        A.append(a_gm_rank)

    allgathered_a = torch.cat(A, dim=0)

    for i in range(args.rank_size):
        b_gm_rank = gen_random_data([K, N], dtype=torch.int8, debug=debug)
        tensor_to_file(b_gm_rank, f"./output/b_gm_rank{i}.bin")
        print(f"Generated data b for rank {i}:")
        print(f"  b_gm_rank{i}.bin: shape={b_gm_rank.shape}")

        # Calculate this rank's contribution to the matmul sum
        accumulator_rank = torch.matmul(allgathered_a.to(torch.float32), b_gm_rank.to(torch.float32))
        accumulator_rank +=  bias_gm.to(torch.float32)
        rank_golden = dequantize(accumulator_rank, per_channel_scale_gm.view(1,N), perTokenScale_gm=per_token_scale_gm, pertensor_scale=True)
        tensor_to_file(rank_golden, f"./output/golden_rank{i}.bin")
        print(f"Generated golden for rank {i}:")
        print(f"  golden_rank{i}.bin: shape={rank_golden.shape}")


    tensor_to_file(d_gm.to(torch.float16), "./output/d_gm.bin")
    tensor_to_file(fused_scale_gm, "./output/scale_gm.bin")
    # tensor_to_file(perTokenScale_gm, "./output/perTokenScale_gm.bin")
    

        
if __name__ == '__main__':
    # os.makedirs(WORKSPACE + "/log", exist_ok=True)
    gen_golden_data()
