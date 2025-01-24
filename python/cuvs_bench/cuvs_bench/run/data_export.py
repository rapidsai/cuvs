#
# Copyright (c) 2024, NVIDIA CORPORATION.
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
#

import json
import os
import traceback

import pandas as pd

skip_build_cols = set(
    [
        "algo_name",
        "index_name",
        "time",
        "name",
        "family_index",
        "per_family_instance_index",
        "run_name",
        "run_type",
        "repetitions",
        "repetition_index",
        "iterations",
        "real_time",
        "time_unit",
        "index_size",
    ]
)

skip_search_cols = (
    set(["recall", "qps", "latency", "items_per_second", "Recall", "Latency"])
    | skip_build_cols
)

metrics = {
    "k-nn": {
        "description": "Recall",
        "worst": float("-inf"),
        "lim": [0.0, 1.03],
    },
    "throughput": {
        "description": "Queries per second (1/s)",
        "worst": float("-inf"),
    },
    "latency": {
        "description": "Search Latency (s)",
        "worst": float("inf"),
    },
}


def read_json_files(dataset, dataset_path, method):
    """
    Yield file paths, algo names, and loaded JSON data as pandas DataFrames.

    Parameters
    ----------
    dataset : str
        The name of the dataset.
    dataset_path : str
        The base path where datasets are stored.
    method : str
        The method subdirectory to search within (e.g., "build" or "search").

    Yields
    ------
    tuple
        A tuple containing the file path, algorithm name, and the
        DataFrame of JSON content.
    """
    dir_path = os.path.join(dataset_path, dataset, "result", method)
    for file in os.listdir(dir_path):
        if file.endswith(".json"):
            file_path = os.path.join(dir_path, file)
            try:
                with open(file_path, "r", encoding="ISO-8859-1") as f:
                    data = json.load(f)
                    df = pd.DataFrame(data["benchmarks"])
                    algo_name = tuple(file.split(",")[:2])
                    yield file_path, algo_name, df
            except Exception as e:
                print(f"Error processing file {file}: {e}. Skipping...")
                traceback.print_exc()


def clean_algo_name(algo_name):
    """
    Clean and format the algorithm name.

    Parameters
    ----------
    algo_name : tuple
        Tuple containing parts of the algorithm name.

    Returns
    -------
    str
        Cleaned algorithm name.
    """

    return algo_name[0] if "base" in algo_name[1] else "_".join(algo_name)


def write_csv(file, algo_name, df, extra_columns=None, skip_cols=None):
    """
    Write a DataFrame to CSV with specified columns skipped.

    Parameters
    ----------
    file : str
        The path to the file to be written.
    algo_name : str
        The algorithm name to be included in the CSV.
    df : pandas.DataFrame
        The DataFrame containing the data to write.
    extra_columns : list, optional
        List of extra columns to add (default is None).
    skip_cols : set, optional
        Set of columns to skip when writing to CSV (default is None).
    """
    df["name"] = df["name"].str.split("/").str[0]
    write_data = pd.DataFrame(
        {
            "algo_name": [algo_name] * len(df),
            "index_name": df["name"],
            "time": df["real_time"],
        }
    )
    # Add extra columns if provided
    if extra_columns:
        for col in extra_columns:
            write_data[col] = None
    # Include columns not in skip list
    for name in df:
        if name not in skip_cols:
            write_data[name] = df[name]
    write_data.to_csv(file.replace(".json", ".csv"), index=False)


def convert_json_to_csv_build(dataset, dataset_path):
    """
    Convert build JSON files to CSV format.

    Parameters
    ----------
    dataset : str
        The name of the dataset.
    dataset_path : str
        The base path where datasets are stored.
    """
    for file, algo_name, df in read_json_files(dataset, dataset_path, "build"):
        try:
            algo_name = clean_algo_name(algo_name)
            write_csv(file, algo_name, df, skip_cols=skip_build_cols)
        except Exception as e:
            print(f"Error processing build file {file}: {e}. Skipping...")
            traceback.print_exc()


def convert_json_to_csv_search(dataset, dataset_path):
    """
    Convert search JSON files to CSV format.

    Parameters
    ----------
    dataset : str
        The name of the dataset.
    dataset_path : str
        The base path where datasets are stored.
    """
    for file, algo_name, df in read_json_files(
        dataset, dataset_path, "search"
    ):
        try:
            build_file = os.path.join(
                dataset_path,
                dataset,
                "result",
                "build",
                f"{','.join(algo_name)}.csv",
            )
            algo_name = clean_algo_name(algo_name)
            df["name"] = df["name"].str.split("/").str[0]
            write = pd.DataFrame(
                {
                    "algo_name": [algo_name] * len(df),
                    "index_name": df["name"],
                    "recall": df["Recall"],
                    "throughput": df["items_per_second"],
                    "latency": df["Latency"],
                }
            )
            # Append build data
            for name in df:
                if name not in skip_search_cols:
                    write[name] = df[name]
            if os.path.exists(build_file):
                build_df = pd.read_csv(build_file)
                write_ncols = len(write.columns)
                write["build time"] = None
                write["build threads"] = None
                write["build cpu_time"] = None
                write["build GPU"] = None

                for col_idx in range(6, len(build_df.columns)):
                    col_name = build_df.columns[col_idx]
                    write[col_name] = None

                for s_index, search_row in write.iterrows():
                    for b_index, build_row in build_df.iterrows():
                        if search_row["index_name"] == build_row["index_name"]:
                            write.iloc[s_index, write_ncols] = build_df.iloc[
                                b_index, 2
                            ]
                            write.iloc[
                                s_index, write_ncols + 1 :
                            ] = build_df.iloc[b_index, 3:]
                            break
            # Write search data and compute frontiers
            write.to_csv(file.replace(".json", ",raw.csv"), index=False)
            write_frontier(file, write, "throughput")
            write_frontier(file, write, "latency")
        except Exception as e:
            print(f"Error processing search file {file}: {e}. Skipping...")
            traceback.print_exc()


def create_pointset(data, xn, yn):
    """
    Create a pointset by sorting and filtering data based on metrics.

    Parameters
    ----------
    data : list
        A list of data points.
    xn : str
        X-axis metric name.
    yn : str
        Y-axis metric name.

    Returns
    -------
    list
        Filtered list of data points sorted by x and y metrics.
    """
    xm, ym = metrics[xn], metrics[yn]
    y_col = 4 if yn == "latency" else 3

    rev_x, rev_y = (-1 if xm["worst"] < 0 else 1), (
        -1 if ym["worst"] < 0 else 1
    )
    # Sort data based on x and y metrics
    data.sort(key=lambda t: (rev_y * t[y_col], rev_x * t[2]))
    lines = []
    last_x = xm["worst"]
    comparator = (
        (lambda xv, lx: xv > lx) if last_x < 0 else (lambda xv, lx: xv < lx)
    )
    for d in data:
        if comparator(d[2], last_x):
            last_x = d[2]
            lines.append(d)
    return lines


def get_frontier(df, metric):
    """
    Get the frontier of the data for a given metric.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    metric : str
        The metric for which to compute the frontier.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the frontier points for the given metric.
    """
    lines = create_pointset(df.values.tolist(), "k-nn", metric)
    return pd.DataFrame(lines, columns=df.columns)


def write_frontier(file, write_data, metric):
    """
    Write the frontier data to CSV for a given metric.

    Parameters
    ----------
    file : str
        The path to the file to write the frontier data.
    write_data : pandas.DataFrame
        The DataFrame containing the original data.
    metric : str
        The metric for which the frontier is computed
        (e.g., "throughput", "latency").
    """
    frontier_data = get_frontier(write_data, metric)
    frontier_data.to_csv(file.replace(".json", f",{metric}.csv"), index=False)
