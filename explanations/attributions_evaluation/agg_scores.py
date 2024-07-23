import glob
import os
import sys

import numpy as np
import pandas as pd

folder_path = "data/b50"

metric = "sensitivity"
file_pattern = f"{metric}*.csv"
attr_name = "vg_grad__ig_grad__sg_grad"


file_paths = glob.glob(os.path.join(folder_path, file_pattern))

all_data = []
data_by_target = {}


def string_to_float(s):
    return float(s.strip("[]"))


for file_path in file_paths:
    if attr_name not in file_path:
        continue

    df = pd.read_csv(file_path)

    if metric in ("sensitivity", "complexity"):
        # only from third column
        df.iloc[:, 2:] = df.iloc[:, 2:].applymap(string_to_float)

    if metric == "time":
        df = df.iloc[[0]]

    all_data.append(df)

    if not metric == "time":
        for target, group in df.groupby("target"):
            if target not in data_by_target:
                data_by_target[target] = []
            data_by_target[target].append(group)

all_data_df = pd.concat(all_data, ignore_index=True)

if metric == "time":
    print(all_data_df.mean())
    sys.exit()
# breakpoint()
means = {col: np.round(all_data_df[col].mean(), 2) for col in all_data_df.columns[2:]}
print(means)

mean_by_target = {}
for target, group_list in data_by_target.items():
    target_df = pd.concat(group_list, ignore_index=True)
    mean_by_target[target] = {
        col: target_df[col].mean() for col in target_df.columns[2:]
    }

print(means, mean_by_target)
