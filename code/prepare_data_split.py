# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import numpy as np


def getSize(PATH):
    count = 0
    dir_path = PATH
    for path in os.scandir(dir_path):
        if path.is_file():
            count += 1
    return count


data_path = "/Users/leo/Desktop/Praktikum/Repo/NVFlare/dataset"
site_num = 2
site_name_prefix = "site-"
size_total = getSize(data_path)#
size_valid = round(0.2 * size_total)
split_method = "uniform"
out_path = "../output"


def data_split_args_parser():
    parser = argparse.ArgumentParser(description="Generate data split for dataset")
    parser.add_argument("--data_path", type=str, help="Path to data file")
    parser.add_argument("--site_num", type=int, help="Total number of sites")
    parser.add_argument("--site_name_prefix", type=str, default="site-", help="Site name prefix")
    parser.add_argument("--size_total", type=int, help="Total number of instances")
    parser.add_argument(
        "--size_valid", type=int, help="Validation size, the first N instances to be treated as validation data"
    )
    parser.add_argument(
        "--split_method",
        type=str,
        default="uniform",
        choices=["uniform", "linear", "square", "exponential"],
        help="How to split the dataset",
    )
    parser.add_argument("--out_path", type=str, default="~/dataset", help="Output path for the data split json file")
    return parser


def split_num_proportion(n, site_num, option: str):
    split = []
    if option == "uniform":
        ratio_vec = np.ones(site_num)
    elif option == "linear":
        ratio_vec = np.linspace(1, site_num, num=site_num)
    elif option == "square":
        ratio_vec = np.square(np.linspace(1, site_num, num=site_num))
    elif option == "exponential":
        ratio_vec = np.exp(np.linspace(1, site_num, num=site_num))
    else:
        raise ValueError("Split method not implemented!")

    total = sum(ratio_vec)
    left = n
    for site in range(site_num - 1):
        x = int(n * ratio_vec[site] / total)
        left = left - x
        split.append(x)
    split.append(left)
    return split


def main():
    #parser = data_split_args_parser()
    #args = parser.parse_args()

    json_data = {"data_path": data_path, "data_index": {"valid": {"start": 0, "end": size_valid}}}

    site_size = split_num_proportion((size_total - size_valid), site_num, split_method)

    for site in range(site_num):
        site_id = site_name_prefix + str(site + 1)
        idx_start = size_valid + sum(site_size[:site])
        idx_end = size_valid + sum(site_size[: site + 1])
        json_data["data_index"][site_id] = {"start": idx_start, "end": idx_end}

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    for site in range(site_num):
        output_file = os.path.join(out_path, f"data_{site_name_prefix}{site + 1}.json")
        with open(output_file, "w") as f:
            json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    main()