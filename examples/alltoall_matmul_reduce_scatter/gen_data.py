import torch
import os
import numpy as np
import argparse

def tensor_to_file(tensor, filename):
    """Write a tensor to a binary file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    tensor_numpy = tensor.cpu().numpy()
    tensor_numpy.tofile(filename)
    print(f"Written tensor with shape {tensor.shape} to {filename}")

def gen_golden_data():
    """
    Generates input data and golden reference for the AlltoallMatmulReduceScatter operator.
    This script should be run once to generate the global data, and then the run.sh
    script will handle creating rank-specific files from the global ones.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int, help="Global M dimension")
    parser.add_argument("n", type=int, help="Global N dimension")
    parser.add_argument("k", type=int, help="Global K dimension")
    parser.add_argument("rank_size", type=int, help="Number of ranks")
    args = parser.parse_args()

    M, N, K, rank_size = args.m, args.n, args.k, args.rank_size
    
    # Check for compatibility
    if M % rank_size != 0 or K % rank_size != 0:
        print(f"Error: M ({M}) and K ({K}) must be divisible by rank_size ({rank_size})")
        exit(1)

    m_local = M // rank_size
    k_local = K // rank_size

    # Use float16 for all data
    dtype = torch.float16

    # 1. Generate Global A and B matrices
    A_global = torch.randn(M, K, dtype=dtype)
    B_global = torch.randn(K, N, dtype=dtype)
    
    tensor_to_file(A_global, f"./output/A_global.bin")
    tensor_to_file(B_global, f"./output/B_global.bin")

    # 2. Calculate Global Golden C
    C_global = torch.matmul(A_global.to(torch.float32), B_global.to(torch.float32)).to(dtype)
    tensor_to_file(C_global, f"./output/C_global.bin")

    # 3. Generate rank-specific data files
    for i in range(rank_size):
        # Slice A for Sequence Parallelism (split along M dim)
        A_local = A_global[i * m_local: (i + 1) * m_local, :]
        
        # Slice B for Tensor Parallelism (split along K dim)
        B_local = B_global[i * k_local: (i + 1) * k_local, :]
        
        # Slice C for golden output (split along M dim)
        C_local_golden = C_global[i * m_local: (i + 1) * m_local, :]

        rank_dir = f"./output/rank_{i}"
        tensor_to_file(A_local, f"{rank_dir}/a_gm.bin")
        tensor_to_file(B_local, f"{rank_dir}/b_gm.bin")
        tensor_to_file(C_local_golden, f"{rank_dir}/golden.bin")

    print("\nData generation complete.")

if __name__ == "__main__":
    gen_golden_data()