import numpy as np


class NPUniformGenerator:
    def __init__(
        self,
        low=-5,
        hi=5,
        output_dtype=np.float16,
    ):
        self.low = low
        self.hi = hi
        self.output_dtype = output_dtype

    def generate(self, shape):
        return np.random.uniform(low=self.low, high=self.hi, size=shape).astype(
            self.output_dtype
        )
