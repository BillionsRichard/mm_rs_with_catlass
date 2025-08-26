import torch
import os
import numpy as np

from utils import tensor_to_file

BIAS_LOW = -65536
BIAS_HIGH = 65536
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
        
def dequantize(golden, scale_gm, perTokenScale_gm):
    golden = golden.to(torch.float32) * (perTokenScale_gm * scale_gm)
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
    # parser.add_argument('rank_id', type = int)
    # parser.add_argument('import_ip', type = int)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    # parser.add_argument('transA', type=int)
    # parser.add_argument('transB', type=int)
    args = parser.parse_args()
    rankSize = args.rank_size
    M, N, K = args.m, args.n, args.k

    debug = int(os.getenv("debug", 0)) == 1
    
    a_gm = gen_random_data([M, K], dtype = torch.int8)
    b_gm = gen_random_data([K, N], dtype = torch.int8)
    scale_gm = torch.empty(size=[1,N], dtype=torch.float32).uniform_(0.004,0.005)

    # use all same value as perTensorScale.
    perTokenScale_gm = torch.full(size=[M,1], fill_value=0.0045, dtype=torch.float32)
    print(f'{perTokenScale_gm=}')
    c_gm = torch.zeros((M * rankSize, N), dtype= torch.int32)
    d_gm = torch.zeros((M * rankSize, N), dtype= torch.float32)
    
    if debug:
        print(f'[warning]: in debug mode, use {torch.arange(1, N+1, dtype=torch.int32)} as bias input.')
        bias_gm = torch.arange(1, N+1, dtype=torch.int32)
    else:
        bias_gm = torch.randint(low=BIAS_LOW, high=BIAS_HIGH+1, size=(N,), dtype=torch.int32)
    
    # a_gm = torch.ones_like(a_gm)
    # b_gm = torch.ones_like(b_gm)
    # scale_gm = torch.tensor(list(range(5,N+5))).to(torch.float32) / 30
    # perTokenScale_gm = torch.tensor(list(range(1,M+1))).to(torch.float32) / 100

    
    matrix_c = torch.matmul(a_gm.to(torch.float32), b_gm.to(torch.float32))
    matrix_c += bias_gm.to(torch.float32)

    tensor_to_file(matrix_c, "./output/c_test.bin")
    golden = dequantize(matrix_c, scale_gm.view(1,N), perTokenScale_gm.view(M,1))
    
    matrix_c_list = []
    for _ in range(rankSize):
        matrix_c_list.append(golden)
    golden = torch.cat(matrix_c_list, dim=0)

    
    tensor_to_file(a_gm, "./output/a_gm.bin")
    tensor_to_file(b_gm, "./output/b_gm.bin")
    tensor_to_file(c_gm, "./output/c_gm.bin")
    tensor_to_file(bias_gm, "./output/bias_gm.bin")

    tensor_to_file(d_gm.to(torch.float16), "./output/d_gm.bin")
    tensor_to_file(scale_gm, "./output/scale_gm.bin")
    tensor_to_file(perTokenScale_gm, "./output/perTokenScale_gm.bin")
    tensor_to_file(golden, "./output/golden.bin")
    
    # compare()
    # if os.path.exists(WORKSPACE + "/output"):
    #     os.system("rm -f {}/output/*".format(WORKSPACE))
        
if __name__ == '__main__':
    # os.makedirs(WORKSPACE + "/log", exist_ok=True)
    gen_golden_data()
