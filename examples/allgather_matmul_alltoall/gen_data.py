import torch
import os
import argparse
import numpy as np

WORKSPACE = os.getcwd()
os.environ["WORKSPACE"] = WORKSPACE

def tensor_to_file(tensor, filename):
    """Write a tensor to a binary file."""
    if tensor.dtype != torch.float16:
        tensor = tensor.to(torch.float16)
    
    # Ensure the output directory exists
    dir_name = os.path.dirname(filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(filename, 'wb') as f:
        f.write(tensor.numpy().tobytes())

def gen_random_data(size, dtype):
    """Generate random data for a given size and dtype."""
    if dtype == torch.float16:
        return torch.randn(size, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def gen_golden_data():
    """Generate input data for each rank and the golden reference output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("rank_size", type=int, help="Number of ranks")
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("output_path", type=str, help="Path to save the generated data")
    args = parser.parse_args()
    
    rank_size = args.rank_size
    m, n, k = args.m, args.n, args.k
    output_path = args.output_path
    
    if n % rank_size != 0:
        raise ValueError("N must be divisible by rank_size")
    
    n_per_rank = n // rank_size
    
    # Lists to hold per-rank tensors for full matrix construction
    a_parts = []
    b_parts = []
    
    print("--- Generating Data for each Rank ---")
    for i in range(rank_size):
        # Generate unique A_i for each rank
        a_i = gen_random_data((m, k), dtype=torch.float16)
        a_parts.append(a_i)
        a_filename = os.path.join(output_path, f"a_gm_rank{i}.bin")
        tensor_to_file(a_i, a_filename)
        print(f"  Rank {i}: Saved A_i to {a_filename} with shape {a_i.shape}")

        # Generate unique B_i for each rank
        b_i = gen_random_data((k, n_per_rank), dtype=torch.float16)
        b_parts.append(b_i)
        b_filename = os.path.join(output_path, f"b_gm_rank{i}.bin")
        tensor_to_file(b_i, b_filename)
        print(f"  Rank {i}: Saved B_i to {b_filename} with shape {b_i.shape}")

    print("\n--- Calculating Golden Reference ---")
    # Construct A_full and B_full on the host for golden calculation
    # A_full is constructed by concatenating A_i matrices along the M dimension
    a_full = torch.cat(a_parts, dim=0)
    # B_full is constructed by concatenating B_i matrices along the N dimension
    b_full = torch.cat(b_parts, dim=1)
    
    print(f"  A_full shape: {a_full.shape}")
    print(f"  B_full shape: {b_full.shape}")

    # Calculate the golden C matrix
    c_golden = torch.matmul(a_full, b_full)
    
    print(f"  C_golden shape: {c_golden.shape}")
    
    # Save the golden C matrix to a file
    golden_filename = os.path.join(output_path, "golden.bin")
    tensor_to_file(c_golden, golden_filename)
    print(f"  Saved C_golden to {golden_filename}")

if __name__ == "__main__":
    gen_golden_data()
