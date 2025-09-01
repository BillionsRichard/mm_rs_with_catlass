import torch
import os
import argparse
import numpy as np

def file_to_tensor(filename, shape, dtype):
    """Read a binary file into a tensor."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    with open(filename, 'rb') as f:
        binary_data = f.read()
    
    np_array = np.frombuffer(binary_data, dtype=dtype.numpy_dtype)
    tensor = torch.from_numpy(np_array).reshape(shape)
    return tensor

def verify_result():
    """Stitch together per-rank outputs and compare with the golden reference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("rank_size", type=int, help="Number of ranks")
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("data_path", type=str, help="Path where data is stored")
    args = parser.parse_args()

    rank_size = args.rank_size
    m, n, k = args.m, args.n, args.k
    data_path = args.data_path

    # 1. Load the golden reference tensor
    golden_shape = (m * rank_size, n)
    golden_filename = os.path.join(data_path, "golden.bin")
    print(f"--- Loading Golden Reference from {golden_filename} ---")
    try:
        c_golden = file_to_tensor(golden_filename, golden_shape, torch.float16)
        print(f"  Loaded C_golden with shape: {c_golden.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False

    # 2. Load all per-rank GPU outputs
    gpu_outputs = []
    print("\n--- Loading GPU Outputs from each Rank ---")
    for i in range(rank_size):
        # The output of the kernel on each rank is [M, N], which is a slice of the full C matrix.
        output_shape = (m, n)
        output_filename = os.path.join(data_path, f"output_rank{i}.bin")
        try:
            c_gpu_i = file_to_tensor(output_filename, output_shape, torch.float16)
            gpu_outputs.append(c_gpu_i)
            print(f"  Loaded Rank {i} output from {output_filename} with shape: {c_gpu_i.shape}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return False

    # 3. Stitch the GPU outputs together to form the full result matrix
    # The per-rank outputs C_gpu_i are slices along the M dimension of the full C matrix.
    c_gpu_full = torch.cat(gpu_outputs, dim=0)
    print(f"\n--- Stitching GPU outputs ---")
    print(f"  Stitched C_gpu_full shape: {c_gpu_full.shape}")

    # 4. Compare the stitched GPU result with the golden reference
    print("\n--- Comparing GPU Output with Golden Reference ---")
    if c_gpu_full.shape != c_golden.shape:
        print(f"{RED}FAIL: Shape mismatch! GPU_full: {c_gpu_full.shape}, Golden: {c_golden.shape}{RESET}")
        return False

    # Using torch.allclose for robust floating point comparison
    are_close = torch.allclose(c_gpu_full.float(), c_golden.float(), rtol=1e-3, atol=1e-5)

    if are_close:
        print(f"\n{GREEN}PASS: Verification successful!{RESET}")
        return True
    else:
        print(f"\n{RED}FAIL: Verification failed!{RESET}")
        diff = torch.abs(c_gpu_full.float() - c_golden.float())
        print(f"  Max difference: {torch.max(diff)}")
        return False

if __name__ == "__main__":
    # Define color codes for terminal output
    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"
    
    try:
        if not verify_result():
            exit(1)
    except Exception as e:
        print(f"{RED}An error occurred: {e}{RESET}")
        exit(1)
