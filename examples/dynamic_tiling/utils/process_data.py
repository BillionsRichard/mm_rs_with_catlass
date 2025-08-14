import os
import pandas as pd
import numpy as np
import subprocess
import time
from typing import List

warm_up_times_str = os.getenv('WARM_UP_TIMES', '10')
WARM_UP_TIMES = int(warm_up_times_str)
perf_test_cycle_times_str = os.getenv('PERF_TEST_CYCLE_TIMES', '3')
PERF_TEST_CYCLE_TIMES = int(perf_test_cycle_times_str)

def open_input_file(input_file):
    df = pd.read_csv(input_file)
    return df

def count_groups_pandas(
    file_path: str,
    key_cols: List[str] = ['M', 'K', 'N']
) -> List[int]:
    df = pd.read_csv(file_path, dtype=str)

    # 标记组边界：当任意 key 列与上行不同时，标记为新组
    diff = (df[key_cols] != df[key_cols].shift(1)).any(axis=1)
    # group_id 为累计的新组序号
    group_id = diff.cumsum()

    # 按 group_id 分组并计数
    counts = df.groupby(group_id).size().tolist()
    return counts

def get_time_data(df, groups: list[int]):
    df = df.reset_index(drop=True)
    time_data = []
    idx = 0
    for coc_tiling_num in groups:
        start_row = idx + WARM_UP_TIMES
        data_rows = WARM_UP_TIMES + coc_tiling_num * PERF_TEST_CYCLE_TIMES
        for i in range(coc_tiling_num):
            current_row = start_row + i * PERF_TEST_CYCLE_TIMES
            group = df.iloc[current_row : current_row + PERF_TEST_CYCLE_TIMES]["task_time(us)"]
            avg_value = group.mean()  # 计算这一组的平均值
            time_data.append(avg_value)
        idx += data_rows
    return time_data

def get_pref_path(path):
    pref_lists = list(filter(lambda item: item.startswith("PROF"), os.listdir(path)))
    data_path_1 = sorted(pref_lists)[-1]
    for data_path in os.listdir(path + "/" + data_path_1):
        if os.path.basename(data_path) != "mindstudio_profiler_output":
            continue
        tmp = path + "/" + data_path_1 + "/" + data_path + "/"
        for f in os.listdir(tmp):
            if os.path.basename(f)[:9] == "task_time":
                res = tmp + f
                return res
    return ""

def find_max_csv_filename(preName, folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv') and f.startswith(preName)]
    
    if not csv_files:
        return None
    
    max_csv = max(csv_files)
    return max_csv

def get_tiling_file(preName, path):
    path = path + "/"
    f = find_max_csv_filename(preName, path)
    file_path = path + f
    return file_path

def process_coc_data(output_path):
    cur_file = os.path.abspath(__file__)
    util_dir = os.path.dirname(cur_file)
    project_root = os.path.dirname(util_dir)
    tiling_path = os.path.join(project_root, 'output', 'tiling')
    tiling_file = get_tiling_file("tilingData", tiling_path)
    tiling_df = open_input_file(tiling_file)
    pref_file = get_pref_path(output_path)
    print(f"Performance data source file: {pref_file}")
    pref_df = open_input_file(pref_file)
    pref_df_filtered = pref_df[~pref_df['kernel_type'].isin(['AI_CORE', 'EVENT_RECORD', 'PROFILING_ENABLE', 'PROFILING_DISABLE'])]
    tiling_groups = count_groups_pandas(tiling_file)
    pref_data = get_time_data(pref_df_filtered, tiling_groups)
    tiling_df['Time(us)'] = pref_data
    tiling_df.to_csv(output_path  + "/result.csv", index=False)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    
    process_coc_data(args.output_path)