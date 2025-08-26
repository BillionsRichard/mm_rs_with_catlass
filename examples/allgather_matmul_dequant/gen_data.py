import torch
import os
import numpy as np

from utils import tensor_to_file

WORKSPACE = os.getcwd()

os.environ["WORKSPACE"] = WORKSPACE

def gen_random_data(size, dtype):
    if dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32:
        return torch.empty(size=size, dtype=dtype).uniform_(-2,2)
    elif dtype == torch.int8:
        return torch.randint(-16, 16, size=size, dtype=dtype)
    else:
        print(f"Invalid dtype: {dtype}.")
        exit(1)
        
def dequantize(golden, fused_scale_gm):
    golden = golden.to(torch.float32) * fused_scale_gm
    return golden

# def compare() -> None:
#     ref = np.fromfile(WORKSPACE + "/output/golden.bin", dtype=np.float32)
#     ret = np.fromfile(WORKSPACE + "/output/output.bin", dtype=np.float16)
#     rdiff = np.abs(ret - ref) / np.abs(ref + 1e-6)
#     precis = len(np.where(rdiff < 0.001)[0]) / len(ref)
#     ret = torch.from_numpy(ret)
#     ref = torch.from_numpy(ref)
#     print(ret)
#     print(ref)
#     print("[SUCCESS]:\n" if torch.allclose(ret, ref, 0.001, 0.001) else "[FAILED]:\n", precis)
#     f = open(WORKSPACE + '/log/ac.log','a')
#     if torch.allclose(ret, ref, 0.001, 0.001):
#         f.write("[SUCCESS]!")
#     else:
#         f.write("[FAILED]!")
#     f.close()
#     return

           
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

    # 1. Generate per-rank A matrices and save them
    a_matrices = []
    for i in range(rankSize):
        a_i = gen_random_data([M, K], dtype=torch.int8)
        a_matrices.append(a_i)
        tensor_to_file(a_i, f"./output/a_gm_rank_{i}.bin")

    # 2. Generate other inputs (B, scales)
    b_gm = gen_random_data([K, N], dtype=torch.int8)
    a_scale = torch.empty(size=[1], dtype=torch.float32).uniform_(0.004, 0.005).item()
    b_scale = torch.empty(size=[1, N], dtype=torch.float32).uniform_(0.004, 0.005)
    
    # 3. Fuse scales on the host
    fused_scale = a_scale * b_scale

    # 4. Create the full A matrix for golden calculation
    a_full = torch.cat(a_matrices, dim=0)

    # 5. Calculate the true golden value
    matrix_c_full = torch.matmul(a_full.to(torch.float32), b_gm.to(torch.float32))
    golden = dequantize(matrix_c_full, fused_scale.view(1, N))

    # 6. Prepare dummy C and D matrices for allocation on device
    c_gm = torch.zeros((M * rankSize, N), dtype=torch.int32)
    d_gm = torch.zeros((M * rankSize, N), dtype=torch.float16)

    # 7. Save all files for the host
    tensor_to_file(b_gm, "./output/b_gm.bin")
    tensor_to_file(c_gm, "./output/c_gm.bin")
    tensor_to_file(d_gm, "./output/d_gm.bin")
    tensor_to_file(fused_scale, "./output/scale_gm.bin")
    tensor_to_file(golden.to(torch.float16), "./output/golden.bin")
    
    # compare()
    # if os.path.exists(WORKSPACE + "/output"):
    #     os.system("rm -f {}/output/*".format(WORKSPACE))
        
if __name__ == '__main__':
    # os.makedirs(WORKSPACE + "/log", exist_ok=True)
    gen_golden_data()
