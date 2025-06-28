import random
from functools import reduce
import torch

# Constraints from requirement doc
SHAPE_DIMS_RANGE = (1, 8)
SHAPE_TOTAL_SIZE_LIMIT = 2**31
SHAPE_DIM_VALUES = [1, 7, 8, 9, 15, 16, 17, 19, 20, 21, 255, 256, 257, 131073]
SHAPE_DIM_RANDOM_RANGE = (1, 1024)
DIST_MEAN_RANGE = (-100, 100)
DIST_STD_RANGE = (1, 25)
OUTLIER_FRACTION = 0.001
OUTLIER_SCALE = {"fp16": 1e-3, "bf16": 1e-3, "fp32": 1e-4}
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _product(factors):
    return reduce(lambda x, y: x * y, factors, 1)


def generate_shapes(num_cases=10):
    """Generates random tensor shapes for matmul based on constraints."""
    generated_shapes = set()
    while len(generated_shapes) < num_cases:
        # For A(..., M, K) and B(..., K, N), C(..., M, N)
        # The first n-2 dims are batch dimensions
        num_batch_dims = random.randint(0, SHAPE_DIMS_RANGE[1] - 2)

        all_dim_values = SHAPE_DIM_VALUES + list(
            range(SHAPE_DIM_RANDOM_RANGE[0], SHAPE_DIM_RANDOM_RANGE[1] + 1)
        )

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
            # Use M, K, N for simplicity in test case
            generated_shapes.add((m, k, n, batch_dims))

    # Return a list of dicts for easier use in parametrize
    from pprint import pprint as pp

    print("generated_shapes:")
    pp(generated_shapes)
    return [
        {"m": m, "k": k, "n": n, "batch_dims": batch_dims}
        for m, k, n, batch_dims in generated_shapes
    ]


def generate_tensor(shape, dtype_str):
    """Generates a tensor with specified distribution and outliers."""
    dtype = DTYPES[dtype_str]
    mean = random.uniform(*DIST_MEAN_RANGE)
    std = random.uniform(*DIST_STD_RANGE)

    tensor = torch.randn(shape) * std + mean

    # Add outliers
    num_elements = tensor.numel()
    num_outliers = int(num_elements * OUTLIER_FRACTION)
    if num_outliers > 0:
        outlier_indices = torch.randint(0, num_elements, (num_outliers,))
        outlier_values = torch.randn(num_outliers) * OUTLIER_SCALE[dtype_str]
        tensor.view(-1)[outlier_indices] = outlier_values

    return tensor.to(dtype)


def get_test_cases(num_cases_per_dtype=5):
    """Generates a list of test cases with all parameter combinations."""
    test_cases = []
    for dtype_str in DTYPES.keys():
        shapes = generate_shapes(num_cases_per_dtype)
        for shape_info in shapes:
            test_cases.append({"dtype": dtype_str, **shape_info})
    return test_cases
