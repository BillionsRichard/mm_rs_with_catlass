# test framework general configs.
import numpy as np
import torch


SHAPE_DIMS_RANGE = (1, 8)
SHAPE_TOTAL_SIZE_LIMIT = 2**31
SHAPE_DIM_VALUES = [1, 7, 8, 9, 15, 16, 17, 19, 20, 21, 255, 256, 257, 131073]
# Reduce the random range for faster test generation
SHAPE_DIM_RANDOM_RANGE = (1, 256)
DIST_MEAN_RANGE = (-5, 5)
MM_AR_OP_CORRECTNESS_INPUT_RANGE = [-5, 5]
NP_RANDOM_SEED=42

DIST_STD_RANGE = (1, 5)
OUTLIER_FRACTION = 0.001
OUTLIER_SCALE = {"fp16": 1e-3, "bf16": 1e-3, "fp32": 1e-4}
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
NUMPY_DTYPES = {
    "fp16": np.float16,
    "bf16": np.float16,
    "fp32": np.float32,
}  # bf16 not in numpy, use fp16 for IO

DTYPE_PRECISIONS = {
    "fp16": (1e-2, 1e-2),
    "bf16": (1e-2, 1e-2),
    "fp32": (1e-3, 1e-3),
}

# 4, 5, 6, 7, 8
SUPPORT_RANKS = [
    2,
    3,
]

CORRECTNESS_TEST_NUM_CASES_PER_DTYPE = 45
NUMERICAL_STABILITY_TEST_NUM_CASES_PER_DTYPE = 5
