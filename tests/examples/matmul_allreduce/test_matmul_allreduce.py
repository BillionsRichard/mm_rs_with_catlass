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

NUM_CASES_PER_DTYPE = 100

# Helper functions (integrated from helper.py)

# Constraints from requirement doc
SHAPE_DIMS_RANGE = (1, 8)
SHAPE_TOTAL_SIZE_LIMIT = 2**31
SHAPE_DIM_VALUES = [1, 7, 8, 9, 15, 16, 17, 19, 20, 21, 255, 256, 257, 131073]
# Reduce the random range for faster test generation
SHAPE_DIM_RANDOM_RANGE = (1, 256)
DIST_MEAN_RANGE = (-100, 100)
DIST_STD_RANGE = (1, 25)
OUTLIER_FRACTION = 0.001
OUTLIER_SCALE = {"fp16": 1e-3, "bf16": 1e-3, "fp32": 1e-4}
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
NUMPY_DTYPES = {
    "fp16": np.float16,
    "bf16": np.float16,
    "fp32": np.float32,
}  # bf16 not in numpy, use fp16 for IO

# Use hardcoded paths as fixtures are not reliable
EXECUTABLE_PATH = "examples/matmul_allreduce/build/matmul_allreduce"
TEST_DATA_DIR = "tests/test_data/matmul_allreduce"


def _product(factors):
    return reduce(lambda x, y: x * y, factors, 1)


def generate_shapes(num_cases=3):
    """Generates random tensor shapes for matmul based on constraints."""
    generated_shapes = set()
    # Limit combinations to avoid excessive generation time
    all_dim_values = SHAPE_DIM_VALUES[:10] + list(
        range(SHAPE_DIM_RANDOM_RANGE[0], SHAPE_DIM_RANDOM_RANGE[1], 64)
    )

    while len(generated_shapes) < num_cases:
        num_batch_dims = random.randint(0, 2)  # Limit batch dims for testing

        batch_dims = tuple(random.choices(all_dim_values, k=num_batch_dims))
        m = random.choice(all_dim_values)
        k = random.choice(all_dim_values)
        n = random.choice(all_dim_values)

        shape_a = batch_dims + (m, k)
        shape_b = batch_dims + (k, n)

        if (
            _product(shape_a) < SHAPE_TOTAL_SIZE_LIMIT
            and _product(shape_b) < SHAPE_TOTAL_SIZE_LIMIT
        ):
            generated_shapes.add((m, k, n, batch_dims))

    return [
        {"m": m, "k": k, "n": n, "batch_dims": batch_dims}
        for m, k, n, batch_dims in generated_shapes
    ]


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
    for dtype_str in ["fp16", "fp32", "bf16"]:
        shapes = generate_shapes(num_cases_per_dtype)
        for shape_info in shapes:
            m = shape_info["m"]
            k = shape_info["k"]
            n = shape_info["n"]
            batch_dims = shape_info["batch_dims"]
            id_str = f"{dtype_str}-w{world_size}-m{m}k{k}n{n}-{batch_dims}"
            # Add world_size parameter
            for world_size in [2, 3, 4, 5, 6, 7, 8]:
                test_cases.append(
                    pytest.param(
                        {"world_size": world_size, 
                         "dtype": dtype_str, 
                         **shape_info},
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
    rank, case_params, ipport, base_device_id, tmp_path, executable_path, test_data_dir
):
    """The function to be executed by each rank's process."""
    world_size = case_params["world_size"]
    m, k, n = case_params["m"], case_params["k"], case_params["n"]
    batch_dims = case_params["batch_dims"]
    dtype_str = case_params["dtype"]

    # Each rank works in its own directory
    rank_dir = tmp_path / f"rank_{rank}"
    out_dir = rank_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    shape_a = batch_dims + (m, k)
    shape_b = batch_dims + (k, n)

    # Generate or load data
    case_hash = hashlib.md5(str(case_params).encode()).hexdigest()
    data_dir = os.path.join(test_data_dir, case_hash)
    a_path = os.path.join(data_dir, f"rank_{rank}_a.bin")
    b_path = os.path.join(data_dir, f"rank_{rank}_b.bin")

    # This function is executed by each process, data must be generated here
    # and saved if persistence is needed.
    A = generate_tensor(shape_a, dtype_str)
    B = generate_tensor(shape_b, dtype_str)

    # Save inputs for the C++ executable
    # The executable expects fp16 for __fp16
    numpy_dtype = NUMPY_DTYPES.get(dtype_str, np.float32)
    with open(out_dir / "a_gm.bin", "wb") as f:
        f.write(A.numpy().astype(numpy_dtype).tobytes())
    with open(out_dir / "b_gm.bin", "wb") as f:
        f.write(B.numpy().astype(numpy_dtype).tobytes())

    # The executable requires a c_gm.bin as well
    shape_c = batch_dims + (m, n)
    C_init = torch.zeros(shape_c, dtype=DTYPES[dtype_str])
    with open(out_dir / "c_gm.bin", "wb") as f:
        f.write(C_init.numpy().astype(numpy_dtype).tobytes())

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
    ]

    # It's better to capture stdout/stderr for debugging
    log_path = rank_dir / "log.txt"
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd, cwd=rank_dir, stdout=log_file, stderr=subprocess.STDOUT
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


@pytest.mark.parametrize("case_params", get_test_cases(num_cases_per_dtype=1))
@pytest.mark.parametrize("world_size", [2, 3, 4, 5, 6, 7, 8])
def test_matmul_allreduce(case_params, world_size, tmp_path):
    """Main test function for matmul_allreduce kernel."""
    if not os.path.exists(EXECUTABLE_PATH):
        pytest.skip(f"Executable not found at {EXECUTABLE_PATH}, run build.sh first.")

    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    world_size = case_params["world_size"]
    m, k, n = case_params["m"], case_params["k"], case_params["n"]
    batch_dims = case_params["batch_dims"]
    dtype_str = case_params["dtype"]
    dtype = DTYPES[dtype_str]
    numpy_dtype = NUMPY_DTYPES.get(dtype_str, np.float32)

    # Setup networking
    master_port = find_free_port()
    master_addr = "127.0.0.1"
    ipport = f"{master_addr}:{master_port}"
    base_device_id = 0

    # Calculate ground truth
    # For reproducibility, let's re-seed before data generation
    random.seed(42)
    torch.manual_seed(42)

    shape_a = batch_dims + (m, k)
    shape_b = batch_dims + (k, n)
    shape_c = batch_dims + (m, n)

    all_A = [generate_tensor(shape_a, dtype_str) for _ in range(world_size)]
    all_B = [generate_tensor(shape_b, dtype_str) for _ in range(world_size)]

    gt_start_time = time.time()
    gt = torch.zeros(shape_c, dtype=dtype)
    for i in range(world_size):
        gt += torch.matmul(all_A[i].float(), all_B[i].float()).to(dtype)
    gt_duration_ms = (time.time() - gt_start_time) * 1000

    # Persist data if required
    if gt_duration_ms > 1.0:
        case_hash = hashlib.md5(str(case_params).encode()).hexdigest()
        data_dir = os.path.join(TEST_DATA_DIR, case_hash)
        os.makedirs(data_dir, exist_ok=True)
        for i in range(world_size):
            with open(os.path.join(data_dir, f"rank_{i}_a.bin"), "wb") as f:
                f.write(all_A[i].numpy().astype(numpy_dtype).tobytes())
            with open(os.path.join(data_dir, f"rank_{i}_b.bin"), "wb") as f:
                f.write(all_B[i].numpy().astype(numpy_dtype).tobytes())
        with open(os.path.join(data_dir, "gt.bin"), "wb") as f:
            f.write(gt.numpy().astype(numpy_dtype).tobytes())

    # Re-seed again for the execution run to use the same data
    random.seed(42)
    torch.manual_seed(42)

    # Run ranks in parallel
    ctx = multiprocessing.get_context("spawn")
    processes = []
    for i in range(world_size):
        p = ctx.Process(
            target=run_matmul_allreduce_kernel,
            args=(
                i,
                case_params,
                ipport,
                base_device_id,
                tmp_path,
                EXECUTABLE_PATH,
                TEST_DATA_DIR,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        assert p.exitcode == 0

    # Verify result from rank 0's output
    output_path = tmp_path / "rank_0/out/output.bin"
    result_data = np.fromfile(output_path, dtype=numpy_dtype)
    result_tensor = torch.from_numpy(result_data).reshape(shape_c).to(dtype)

    # Precision thresholds might need adjustment based on dtype
    rtol, atol = 1e-4, 1e-3
    assert torch.allclose(result_tensor.float(), gt.float(), rtol=rtol, atol=atol)
