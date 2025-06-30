import multiprocessing
import time
import pytest
import torch
import numpy as np
import os
import subprocess
import socket
from contextlib import closing
import hashlib
import random
from functools import reduce

# import tests greneral configs.
from tests.examples.config import SHAPE_TOTAL_SIZE_LIMIT
from tests.examples.config import SHAPE_DIM_VALUES
from tests.examples.config import SHAPE_DIM_RANDOM_RANGE
from tests.examples.config import DIST_MEAN_RANGE
from tests.examples.config import DIST_STD_RANGE
from tests.examples.config import OUTLIER_FRACTION
from tests.examples.config import OUTLIER_SCALE
from tests.examples.config import DTYPES
from tests.examples.config import NUMPY_DTYPES
from tests.examples.config import DTYPE_PRECISIONS
from tests.examples.config import SUPPORT_RANKS
from tests.examples.config import NUM_CASES_PER_DTYPE

# Use hardcoded paths as fixtures are not reliable
EXECUTABLE_PATH = os.path.abspath("./examples/matmul_allreduce/out/matmul_allreduce")
print(f"{EXECUTABLE_PATH=}")
TEST_DATA_DIR = "tests/examples/matmul_allreduce/test_data"


def _product(factors):
    return reduce(lambda x, y: x * y, factors, 1)


def generate_shapes(num_cases=1):
    """Generates random tensor shapes for matmul based on constraints."""
    generated_shapes = set()
    # Limit combinations to avoid excessive generation time
    all_dim_values = SHAPE_DIM_VALUES[:10] + list(
        range(SHAPE_DIM_RANDOM_RANGE[0], SHAPE_DIM_RANDOM_RANGE[1], 64)
    )

    while len(generated_shapes) < num_cases:
        # num_batch_dims = random.randint(1, 3)  # Limit batch dims for testing

        # batch_dims = tuple(random.choices(all_dim_values))
        m = random.choice(all_dim_values)
        k = random.choice(all_dim_values)
        n = random.choice(all_dim_values)

        shape_a = (m, k)
        shape_b = (k, n)

        if (
            _product(shape_a) < SHAPE_TOTAL_SIZE_LIMIT
            and _product(shape_b) < SHAPE_TOTAL_SIZE_LIMIT
        ):
            generated_shapes.add((m, k, n))

    return [{"m": m, "k": k, "n": n} for m, k, n in generated_shapes]


def generate_tensor(shape, dtype_str):
    """Generates a tensor with specified distribution and outliers."""
    dtype = DTYPES[dtype_str]
    mean = random.uniform(*DIST_MEAN_RANGE)
    std = random.uniform(*DIST_STD_RANGE)

    tensor = torch.randn(shape, dtype=torch.float32) * std + mean

    num_elements = tensor.numel()
    num_outliers = int(num_elements * OUTLIER_FRACTION)
    if num_outliers > 0:
        outlier_indices = torch.randint(0, num_elements, (num_outliers,))
        outlier_values = torch.randn(num_outliers) * OUTLIER_SCALE[dtype_str]
        tensor.view(-1)[outlier_indices] = outlier_values.to(torch.float32)

    return tensor.to(dtype)


def get_test_cases(num_cases_per_dtype=NUM_CASES_PER_DTYPE):
    """Generates a list of test cases."""
    test_cases = []
    # Limit dtypes for faster testing
    # , "fp32", "bf16"
    for dtype_str in ["fp16"]:
        shapes = generate_shapes(num_cases_per_dtype)
        for shape_info in shapes:
            m = shape_info["m"]
            k = shape_info["k"]
            n = shape_info["n"]
            # batch_dims = shape_info["batch_dims"]
            # Add world_size parameter
            for world_size in SUPPORT_RANKS:
                id_str = f"{dtype_str}-w{world_size}-m{m}k{k}n{n}"
                test_cases.append(
                    pytest.param(
                        {"world_size": world_size, "dtype": dtype_str, **shape_info},
                        id=id_str,
                    )
                )
    return test_cases


# Test implementation
def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def run_matmul_allreduce_kernel(
    rank, case_params, ipport, base_device_id, executable_path, test_data_dir
):
    """The function to be executed by each rank's process."""
    world_size = case_params["world_size"]
    m, k, n = case_params["m"], case_params["k"], case_params["n"]

    # Launch the C++ executable
    cmd = [
        executable_path,
        str(world_size),
        str(rank),
        ipport,
        str(base_device_id),
        str(m),
        str(k),
        str(n),
        test_data_dir,
    ]

    # It's better to capture stdout/stderr for debugging
    log_path = os.path.join(test_data_dir, "log.txt")
    print(f"{cmd=}, {os.getcwd()=}")
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd, cwd=test_data_dir, stdout=log_file, stderr=subprocess.STDOUT
        )
        proc.wait()

    if proc.returncode != 0:
        # This allows pytest to show the logs on failure
        with open(log_path, "r") as f:
            print(f"--- RANK {rank} LOGS ---")
            print(f.read())
        pytest.fail(
            f"Rank {rank} failed with exit code {proc.returncode}", pytrace=False
        )


@pytest.mark.parametrize(
    "case_params", get_test_cases(num_cases_per_dtype=NUM_CASES_PER_DTYPE)
)
def test_matmul_allreduce(case_params):
    """Main test function for matmul_allreduce kernel."""
    if not os.path.exists(EXECUTABLE_PATH):
        pytest.skip(f"Executable not found at {EXECUTABLE_PATH}, run build.sh first.")

    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    world_size = case_params["world_size"]
    m, k, n = case_params["m"], case_params["k"], case_params["n"]
    # batch_dims = case_params["batch_dims"]
    dtype_str = case_params["dtype"]
    dtype = DTYPES[dtype_str]
    numpy_dtype = NUMPY_DTYPES.get(dtype_str, np.float32)

    # Setup networking
    master_port = find_free_port()
    master_addr = "127.0.0.1"
    ipport = f"tcp://{master_addr}:{master_port}"
    base_device_id = 0

    # Calculate ground truth
    # For reproducibility, let's re-seed before data generation
    random.seed(42)
    torch.manual_seed(42)

    shape_a = (m, k)
    shape_b = (k, n)
    shape_c = (m, n)

    all_A = [generate_tensor(shape_a, dtype_str) for _ in range(world_size)]
    all_B = [generate_tensor(shape_b, dtype_str) for _ in range(world_size)]

    gt_start_time = time.time()
    # cal CPU matmul & allreduce.
    gt_fp32 = torch.zeros(shape_c, dtype=torch.float32)
    for i in range(world_size):
        gt_fp32 += torch.matmul(all_A[i].float(), all_B[i].float())
    gt = gt_fp32.to(dtype)

    # Check for overflow in ground truth calculation and skip if it occurs
    if torch.isinf(gt).any() or torch.isnan(gt).any():
        case_id_str = f"{dtype_str}-w{world_size}-m{m}k{k}n{n}"
        print(
            f"\nINFO: Overflow detected during ground truth calculation for case {case_id_str}. Skipping test case."
        )
        pytest.skip("Skipping test due to overflow in ground truth generation.")

    gt_duration_ms = (time.time() - gt_start_time) * 1000

    # Persist data if required
    # if gt_duration_ms > 1.0:
    case_hash = hashlib.md5(str(case_params).encode()).hexdigest()
    case_params["case_id"] = case_hash
    # data_dir is independent for every single test case.
    data_dir = os.path.abspath(os.path.join(TEST_DATA_DIR, case_hash))
    print(f"{data_dir=}")
    os.makedirs(data_dir, exist_ok=True)
    for i in range(world_size):
        rank_i_a_path = os.path.abspath(os.path.join(data_dir, f"rank_{i}_a.bin"))
        rank_i_b_path = os.path.abspath(os.path.join(data_dir, f"rank_{i}_b.bin"))
        # print(f"{rank_i_a_path=}")
        # print(f"{rank_i_b_path=}")
        with open(rank_i_a_path, "wb") as f:
            f.write(all_A[i].numpy().astype(numpy_dtype).tobytes())

        with open(rank_i_b_path, "wb") as f:
            f.write(all_B[i].numpy().astype(numpy_dtype).tobytes())

    with open(os.path.join(data_dir, "gt.bin"), "wb") as f:
        f.write(gt.numpy().astype(numpy_dtype).tobytes())

    # pack CPU input & output.
    case_params[case_hash] = {"A": all_A[i], "B": all_B[i], "gt": gt}

    # Re-seed again for the execution run to use the same data
    random.seed(42)
    torch.manual_seed(42)

    # Run ranks in parallel
    ctx = multiprocessing.get_context("spawn")
    processes = []
    for rank_id in range(world_size):
        p = ctx.Process(
            target=run_matmul_allreduce_kernel,
            args=(
                rank_id,
                case_params,
                ipport,
                base_device_id,
                EXECUTABLE_PATH,
                data_dir,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        assert p.exitcode == 0

    # Verify result from rank 0's output
    output_path = os.path.join(data_dir, "shmem_output.bin")
    # print(f'shmem matmul_allreduce resultt file: {os.path.abspath(output_path)=}')
    result_data = np.fromfile(output_path, dtype=numpy_dtype)
    print(f"{shape_a=}, {shape_b=}, {shape_c=}")
    np_result = torch.from_numpy(result_data)
    # print(f'{np_result.shape=}')
    result_tensor = np_result.reshape(shape_c).to(dtype)

    # # Robust assertion for fp16, handling NaN and Inf
    # # 1. Check for NaNs
    # nan_mask_result = torch.isnan(result_tensor)
    # nan_mask_gt = torch.isnan(gt)
    # assert torch.equal(nan_mask_result, nan_mask_gt), "Mismatch in NaN values"

    # # 2. Check for Infs
    # inf_mask_result = torch.isinf(result_tensor)
    # inf_mask_gt = torch.isinf(gt)
    # assert torch.equal(inf_mask_result, inf_mask_gt), "Mismatch in Inf values"

    # # 3. Compare finite values
    # finite_mask = ~nan_mask_result & ~inf_mask_result
    rtol, atol = DTYPE_PRECISIONS.get(dtype_str, (1e-2, 1e-2))
    assert torch.allclose(result_tensor, gt, rtol=rtol, atol=atol), f"{output_path=}"
