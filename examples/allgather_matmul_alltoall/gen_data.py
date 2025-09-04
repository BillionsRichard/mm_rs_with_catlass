import numpy as np
import sys
import os

def gen_and_save_data(M, N, K, rank_size, dtype=np.float16):
    print(f"Generating data for M={M}, N={N}, K={K}, rank_size={rank_size}")

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if N % rank_size != 0:
        raise ValueError("N must be divisible by rank_size")
    
    N_per_rank = N // rank_size

    # 1. Generate Inputs for each rank
    all_A = [np.random.randn(M, K).astype(dtype) for _ in range(rank_size)]
    all_B = [np.random.randn(K, N_per_rank).astype(dtype) for _ in range(rank_size)]

    for i in range(rank_size):
        all_A[i].tofile(os.path.join(output_dir, f"a_gm_rank{i}.bin"))
        all_B[i].tofile(os.path.join(output_dir, f"b_gm_rank{i}.bin"))
    print(f"Saved input files for {rank_size} ranks.")

    # 2. Simulate the Matmul-Scatter logic to get the final golden result
    
    # Step 2.1: Simulate the Allgather of A
    # Result is a single tensor of shape [rank_size, M, K]
    A_gathered = np.stack(all_A, axis=0)

    # Step 2.2: Simulate the computation on each rank
    # Each rank `i` computes `A_gathered @ B_i`
    all_C_partial = []
    for i in range(rank_size):
        B_i = all_B[i]
        # Reshape B for batched matmul broadcasting
        # (rank_size, M, K) @ (1, K, N_per_rank) -> (rank_size, M, N_per_rank)
        C_partial_i = A_gathered @ B_i.reshape(1, K, N_per_rank)
        all_C_partial.append(C_partial_i)

    # Step 2.3: Simulate the Scatter/Alltoall result
    # The final output on rank `j` is formed by taking the j-th slice
    # from every C_partial_i and concatenating them.
    for j in range(rank_size):
        # Slices for rank j: [C_partial_0[j], C_partial_1[j], ..., C_partial_{rank_size-1}[j]]
        # Each slice has shape (M, N_per_rank)
        slices_for_j = [all_C_partial[i][j] for i in range(rank_size)]
        
        # Concatenate along the N dimension
        C_golden_j = np.concatenate(slices_for_j, axis=1)
        
        if C_golden_j.shape != (M, N):
             raise ValueError(f"Shape mismatch for golden output on rank {j}. Expected {(M,N)}, got {C_golden_j.shape}")

        C_golden_j.tofile(os.path.join(output_dir, f"golden_rank{j}.bin"))
        print(f"Saved golden_rank{j}.bin with shape {C_golden_j.shape}")

    print("Successfully generated all input and golden files based on Matmul-Scatter logic.")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python gen_data.py <M> <N> <K> <rank_size>")
        sys.exit(1)
    
    M = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    rank_size = int(sys.argv[4])
    
    gen_and_save_data(M, N, K, rank_size)
