#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import csv
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

def _row_from_benchmark_entry(algo_name, entry):
    """Build a single CSV row dict from one benchmark entry (for live append)."""
    name = entry.get("name") or entry.get("run_name") or ""
    index_name = str(name).split("/")[0]
    recall = entry.get("Recall") or entry.get("recall")
    throughput = entry.get("items_per_second") or entry.get("qps")
    latency = entry.get("Latency") or entry.get("latency")
    row = {
        "algo_name": algo_name,
        "index_name": index_name,
        "recall": recall,
        "throughput": throughput,
        "latency": latency,
    }
    for k, v in entry.items():
        if k not in skip_search_cols:
            row["search_label" if k == "label" else k] = v
    return row


def append_search_row_to_csv(csv_path, algo_name, benchmark_entry):
    """
    Append one benchmark result as a row to a CSV file (for --live-csv).
    Creates the file with header on first call, then appends rows.
    """
    row = _row_from_benchmark_entry(algo_name, benchmark_entry)
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    if not file_exists:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
    else:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writerow({k: row.get(k) for k in header})

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
    if not os.path.isdir(dir_path):
        return
    for file in os.listdir(dir_path):
        if file.endswith(".json") and not file.startswith("_live_"):
            file_path = os.path.join(dir_path, file)
            try:
                with open(file_path, "r", encoding="ISO-8859-1") as f:
                    data = json.load(f)
                    df = pd.DataFrame(data["benchmarks"])
                    base = file.replace(".json", "")
                    if "," in base:
                        algo_name = tuple(base.split(",")[:2])
                    else:
                        parts = base.split("_")
                        algo_name = (parts[0], parts[1]) if len(parts) >= 2 else (parts[0],)
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

    name = algo_name[0] if "base" in algo_name[1] else "_".join(algo_name)
    return name.removesuffix(".json")


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
    if df.empty or len(df.columns) == 0:
        raise ValueError("empty benchmarks (no rows or no columns)")
    # "name" / "run_name": benchmark identifier from Google Benchmark (e.g. full run name
    # like "cuvs_ivf_pq.nlist1024.pq_dim64.../0/process_time/real_time"). We take the part
    # before the first "/" as index_name (the config id) for the CSV.
    name_col = "name" if "name" in df.columns else "run_name"
    if name_col not in df.columns:
        raise KeyError(
            f"Build JSON must contain 'name' or 'run_name'; got columns: {list(df.columns)}"
        )
    index_name = df[name_col].astype(str).str.split("/").str[0]
    time_col = "real_time" if "real_time" in df.columns else "cpu_time"
    if time_col not in df.columns:
        raise KeyError(
            f"Build JSON must contain 'real_time' or 'cpu_time'; got columns: {list(df.columns)}"
        )
    write_data = pd.DataFrame(
        {
            "algo_name": [algo_name] * len(df),
            "index_name": index_name,
            "time": df[time_col],
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
    out_path = os.path.abspath(file.replace(".json", ".csv"))
    write_data.to_csv(out_path, index=False)
    print(f"[cuvs_bench] Wrote build CSV: {out_path}")


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
    dir_path = os.path.join(dataset_path, dataset, "result", "build")
    dir_path_abs = os.path.abspath(dir_path)
    print(f"[cuvs_bench] CSV export: looking for build JSON in {dir_path_abs}")
    if not os.path.isdir(dir_path):
        print(f"[cuvs_bench] No result/build dir at {dir_path_abs}; skipping CSV export.")
        return
    count = 0
    for file, algo_name, df in read_json_files(dataset, dataset_path, "build"):
        try:
            algo_name = clean_algo_name(algo_name)
            write_csv(file, algo_name, df, skip_cols=skip_build_cols)
            count += 1
        except Exception as e:
            print(f"Error processing build file {file}: {e}. Skipping...")
            traceback.print_exc()
    if count == 0 and os.path.isdir(dir_path):
        print(f"[cuvs_bench] No .json files in {dir_path_abs}; no build CSV written.")


def _get_column(df, *candidates):
    """Return the first column name in df that exists, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


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
    dir_path = os.path.join(dataset_path, dataset, "result", "search")
    dir_path_abs = os.path.abspath(dir_path)
    print(f"[cuvs_bench] CSV export: looking for search JSON in {dir_path_abs}")
    if not os.path.isdir(dir_path):
        print(f"[cuvs_bench] No result/search dir at {dir_path_abs}; skipping CSV export.")
        return
    count = 0
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
            if not os.path.exists(build_file):
                build_file_alt = os.path.join(
                    dataset_path,
                    dataset,
                    "result",
                    "build",
                    f"{'_'.join(algo_name)}.csv",
                )
                if os.path.exists(build_file_alt):
                    build_file = build_file_alt
            algo_name = clean_algo_name(algo_name)
            name_col = "name" if "name" in df.columns else "run_name"
            if name_col not in df.columns:
                raise KeyError(
                    f"Search JSON must contain 'name' or 'run_name'; got {list(df.columns)}"
                )
            index_name = df[name_col].astype(str).str.split("/").str[0]
            # Required metrics (Google Benchmark counter names)
            recall_col = "Recall" if "Recall" in df.columns else "recall"
            qps_col = "items_per_second" if "items_per_second" in df.columns else "qps"
            lat_col = "Latency" if "Latency" in df.columns else "latency"
            for c in (recall_col, qps_col, lat_col):
                if c not in df.columns:
                    raise KeyError(
                        f"Search JSON must contain recall, throughput, latency; got {list(df.columns)}"
                    )
            write = pd.DataFrame(
                {
                    "algo_name": [algo_name] * len(df),
                    "index_name": index_name,
                    "recall": df[recall_col],
                    "throughput": df[qps_col],
                    "latency": df[lat_col],
                }
            )
            # Append build data
            for name in df:
                if name not in skip_search_cols:
                    # distinguish search label from build label
                    write["search_label" if name == "label" else name] = df[
                        name
                    ]
            if os.path.exists(build_file):
                build_df = pd.read_csv(build_file)
                write_ncols = len(write.columns)
                write["build time"] = None
                write["build threads"] = None
                write["build cpu_time"] = None

                start_idx = 5
                if "GPU" in build_df.columns:
                    start_idx = 6
                    write["build GPU"] = None
                for col_idx in range(start_idx, len(build_df.columns)):
                    col_name = build_df.columns[col_idx]
                    write[col_name] = None
                    if col_name == "num_threads":
                        write["build_num_threads"] = None
                for s_index, search_row in write.iterrows():
                    for b_index, build_row in build_df.iterrows():
                        if search_row["index_name"] == build_row["index_name"]:
                            write.iloc[s_index, write_ncols] = build_df.iloc[
                                b_index, 2
                            ]
                            write.iloc[s_index, write_ncols + 1 :] = (
                                build_df.iloc[b_index, 3:]
                            )
                            break
            # Write search data and compute frontiers
            raw_path = os.path.abspath(file.replace(".json", ",raw.csv"))
            write.to_csv(raw_path, index=False)
            print(f"[cuvs_bench] Wrote search CSV: {raw_path}")
            write_frontier(file, write, "throughput")
            write_frontier(file, write, "latency")
            count += 1
        except Exception as e:
            print(f"Error processing search file {file}: {e}. Skipping...")
            traceback.print_exc()
    if count == 0 and os.path.isdir(dir_path):
        print(f"[cuvs_bench] No .json files in {dir_path_abs} or conversion failed; no search CSV written.")

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

    rev_x, rev_y = (
        (-1 if xm["worst"] < 0 else 1),
        (-1 if ym["worst"] < 0 else 1),
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
