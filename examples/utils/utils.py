from enum import IntEnum
import numpy as np
import torch

class CommType(IntEnum):
    MATMUL_ALLREDUCE = 0
    ALLGATHER_MATMUL = 1
    MATMUL_REDUCE_SCATTER = 2
    ALLGATHER_MATMUL_WITH_GATHER_RESULT = 4

    @classmethod
    def from_str(cls, arg: str):
        return cls(int(arg))

class DataType(IntEnum):
    FLOAT = 0
    FLOAT16 = 1
    INT8 = 2
    BF16 = 27

    @classmethod
    def from_str(cls, arg: str):
        return cls(int(arg))

    @property
    def torch_type(self):
        return {
            DataType.FLOAT: torch.float,
            DataType.FLOAT16: torch.float16,
            DataType.INT8: torch.int8,
            DataType.BF16: torch.bfloat16,
        }[self]
    @property
    def numpy_type(self):
        return {
            DataType.FLOAT: np.float32,
            DataType.FLOAT16: np.float16,
            DataType.INT8: np.int8,
            DataType.BF16: None,
        }[self]

def tensor_to_file(tensor: torch.Tensor, file_name: str) -> None:
    if tensor.dtype == torch.bfloat16:
        tensor.view(torch.uint16).numpy().tofile(file_name)
    else:
        tensor.numpy().tofile(file_name)

def tensor_from_file(file_name: str, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.bfloat16:
        return torch.from_numpy(np.fromfile(file_name, dtype=np.float16)).view(torch.bfloat16)
    else:
        numpy_dtype = torch.empty(0, dtype=dtype).numpy().dtype
        return torch.from_numpy(np.fromfile(file_name, numpy_dtype))

def get_rtol(dtype: torch.dtype, compute_times: int) -> float:
    if dtype == torch.float16:
        return 2 ** (-8) if compute_times < 2048 else 2 ** (-7)
    elif dtype == torch.bfloat16:
        return 2 ** (-7) if compute_times < 2048 else 2 ** (-6)
    elif dtype == torch.float32:
        return 2 ** (-11) if compute_times < 2048 else 2 ** (-10)
    else:
        raise ValueError(f"Invalid dtype: {dtype}.")