import numpy as np
import os

from ml_dtypes import bfloat16

def gen_random_data(size, dtype):
    return np.random.uniform(low=0.0, high=10.0, size=size).astype(dtype)

def golden_generate(data_len, rank_size, data_type):
    golden_dir = f"allgather_{data_len}_{rank_size}"
    cmd = f"mkdir golden/{golden_dir}"
    os.system(cmd)

    input_gm = np.zeros((rank_size, data_len), dtype=data_type)
    output_gm = np.zeros((rank_size * data_len), dtype=data_type)

    for i in range(rank_size):
        input_gm[i][:] = gen_random_data((data_len), dtype=data_type)
        output_gm[i * data_len : i * data_len + data_len] = input_gm[i]
    
    for i in range(rank_size):
        input_gm[i].tofile(f"./golden/{golden_dir}/input_gm_{i}.bin")
    output_gm.tofile(f"./golden/{golden_dir}/golden.bin")
    print(f"{data_len} golden generate success !")

def gen_golden_data():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('rank_size', type=int)
    parser.add_argument('test_type', type=str)
    args = parser.parse_args()

    type_map = {
        "int" : np.int32,
        "int32_t" : np.int32,
        "float16_t" : np.float16,
        "bfloat16_t" : bfloat16
    }

    data_type = type_map[args.test_type]
    rank_size = args.rank_size

    case_num = 24
    for i in range(case_num):
        data_len = 16 * (2 ** i)
        golden_generate(data_len, rank_size, data_type)

if __name__ == '__main__':
    gen_golden_data()