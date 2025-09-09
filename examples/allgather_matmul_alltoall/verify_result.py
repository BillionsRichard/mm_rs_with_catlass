import torch
import os
import argparse
import numpy as np
from utils import DataType

def file_to_tensor(filename, shape, dtype):
    """Read a binary file into a tensor."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    with open(filename, 'rb') as f:
        binary_data = f.read()
    
    np_array = np.frombuffer(binary_data, dtype=dtype)
    tensor = torch.from_numpy(np_array).reshape(shape)
    return tensor

def verify_result():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('rank_size', type=int)
    parser.add_argument('output', type=str)
    parser.add_argument('golden', type=str)
    parser.add_argument('out_dtype', type=DataType.from_str, choices=[DataType.FLOAT16, DataType.BF16])
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)

    args = parser.parse_args()

    rank_size = args.rank_size
    m, n, k = args.m, args.n, args.k
    # data_path = args.data_path

    # 1. Load the golden reference tensor
    golden_shape = (m, n)
    golden_filename = args.golden
    print(f"--- Loading Golden Reference from {golden_filename} ---")
    try:
        c_golden = file_to_tensor(golden_filename, golden_shape, np.float16)
        print(f"  Loaded C_golden with shape: {c_golden.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False

    # 2. Load all per-rank NPU outputs
    # npu_outputs = []
    output_file = args.output
    print("\n--- Loading NPU Outputs from each Rank ---")
    try:
        c_npu_i = file_to_tensor(args.output, golden_shape, np.float16)
        # npu_outputs.append(c_npu_i)
        print(f"  Loaded output from {output_file} with shape: {golden_shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False

    # 3. Stitch the NPU outputs together to form the full result matrix
    # The per-rank outputs C_npu_i are slices along the M dimension of the full C matrix.
    # c_npu_full = torch.cat(npu_outputs, dim=0)
    # print(f"\n--- Stitching NPU outputs ---")
    # print(f"  Stitched C_npu_full shape: {c_npu_full.shape}")

    # 4. Compare the stitched NPU result with the golden reference
    print("\n--- Comparing NPU Output with Golden Reference ---")
    if c_npu_i.shape != c_golden.shape:
        print(f"{RED}FAIL: Shape mismatch! NPU_full: {c_npu_i.shape}, Golden: {c_golden.shape}{RESET}")
        return False

    # Using torch.allclose for robust floating point comparison
    are_close = torch.allclose(c_npu_i.float(), c_golden.float(), rtol=1e-3, atol=1e-5)

    if are_close:
        print(f"\n{GREEN}PASS: Verification successful!{RESET}")
        return True
    else:
        print(f"\n{RED}FAIL: Verification failed!{RESET}")
        diff = torch.abs(c_npu_i.float() - c_golden.float())
        print(f'c_golden=\n{c_golden}')
        print(f'c_npu_i=\n{c_npu_i}')
        print(f"  Max difference: {torch.max(diff)}")
        return False

if __name__ == "__main__":
    # Define color codes for terminal output
    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"
    print('Verify begin.....')
    try:
        if not verify_result():
            exit(1)
    except Exception as e:
        print(f"{RED}An error occurred: {e}{RESET}")
        exit(1)
