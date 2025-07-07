import os
import pandas as pd
import numpy as np
import subprocess
import time

WARM_UP_TIMES = 10
PERF_TEST_CYCLE_TIMES = 40

def open_input_file(input_file):
    df = pd.read_csv(input_file)
    return df

def get_time_data(df, testLineNum: int):
    df = df[df['kernel_type'] == "KERNEL_MIX_AIC"]
    df = df.reset_index(drop=True)
    time_data = []
    total_rows = len(df)
    data_rows = total_rows // testLineNum
    coc_tiling_num = (data_rows - WARM_UP_TIMES) // PERF_TEST_CYCLE_TIMES

    for i in range(testLineNum):
        start_row = i * data_rows + WARM_UP_TIMES
        for j in range(coc_tiling_num):
            current_row = start_row + j * PERF_TEST_CYCLE_TIMES
            group = df.iloc[current_row : current_row + PERF_TEST_CYCLE_TIMES]["task_time(us)"]
            avg_value = group.mean()
            time_data.append(avg_value)

    return time_data

def get_time_file(path):
    for data_path in os.listdir(path):
        if os.path.basename(data_path) != "mindstudio_profiler_output":
            continue
        profiler_path = os.path.join(path, data_path)
        for f in os.listdir(profiler_path):
            if os.path.basename(f)[:9] == "task_time":
                res = os.path.join(profiler_path, f)
                return res
    return ""

def get_pref_path(path):
    perf_list = list(filter(lambda item: item.startswith("PROF"), os.listdir(path)))
    print(perf_list)
    task_time_list = []
    for perf_dir in perf_list:
        perf_path = os.path.join(path, perf_dir)
        time_file = get_time_file(perf_path)
        task_time_list.append(time_file)

    return task_time_list


def process_kernel_data():
    tiling_df = open_input_file("../../build/bin/results.csv")
    print(tiling_df)

    pref_file_list = get_pref_path("./output")
    print(pref_file_list)

    case_num = len(tiling_df)
    perf_output = np.zeros((case_num, )).astype(np.float32)

    # Average x Cards' data of all cases
    for pref_file in pref_file_list:
        pref_df = open_input_file(pref_file)
        pref_data = get_time_data(pref_df, case_num)
        perf_output += np.array(pref_data).astype(np.float32)

    perf_output = perf_output / len(pref_file_list)
    tiling_df['Time(us)'] = perf_output
    tiling_df.to_csv("./result.csv", index=False)

if __name__ == '__main__':
    process_kernel_data()
