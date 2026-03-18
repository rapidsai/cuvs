#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# This script is inspired by
# 1: https://github.com/erikbern/ann-benchmarks/blob/main/plot.py
# 2: https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/plotting/utils.py  # noqa: E501
# 3: https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/plotting/metrics.py  # noqa: E501
# License: https://github.com/rapidsai/cuvs/blob/main/thirdparty/LICENSES/LICENSE.ann-benchmark # noqa: E501

import itertools
import os
import re
from collections import OrderedDict

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")

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


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise click.BadParameter(f"{value} is not a positive integer")
    return ivalue


def positive_float(value):
    fvalue = float(value)
    if fvalue <= 0:
        raise click.BadParameter(f"{value} is not a positive float")
    return fvalue


def generate_n_colors(n):
    vs = np.linspace(0.3, 0.9, 7)
    colors = [(0.9, 0.4, 0.4, 1.0)]

    def euclidean(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    while len(colors) < n:
        new_color = max(
            itertools.product(vs, vs, vs),
            key=lambda a: min(euclidean(a, b) for b in colors),
        )
        colors.append(new_color + (1.0,))
    return colors


def create_linestyles(unique_algorithms):
    colors = dict(
        zip(unique_algorithms, generate_n_colors(len(unique_algorithms)))
    )
    linestyles = dict(
        (algo, ["--", "-.", "-", ":"][i % 4])
        for i, algo in enumerate(unique_algorithms)
    )
    markerstyles = dict(
        (algo, ["+", "<", "o", "*", "x"][i % 5])
        for i, algo in enumerate(unique_algorithms)
    )
    faded = dict(
        (algo, (r, g, b, 0.3)) for algo, (r, g, b, a) in colors.items()
    )
    return dict(
        (
            algo,
            (colors[algo], faded[algo], linestyles[algo], markerstyles[algo]),
        )
        for algo in unique_algorithms
    )


def create_plot_search(
    all_data,
    x_scale,
    y_scale,
    fn_out,
    linestyles,
    dataset,
    k,
    batch_size,
    mode,
    time_unit,
    x_start,
    y_max=None,
    x_max=None,
):
    xn = "k-nn"
    xm, ym = (metrics[xn], metrics[mode])
    xm["lim"][0] = x_start
    # Now generate each plot
    handles = []
    labels = []
    plt.figure(figsize=(12, 9))

    # Sorting by mean y-value helps aligning plots with labels
    def mean_y(algo):
        points = np.array(all_data[algo], dtype=object)
        return -np.log(np.array(points[:, 3], dtype=np.float32)).mean()

    # Find range for logit x-scale and for throughput y-scale
    min_x, max_x = 1, 0
    max_y_data = 0
    for algo in sorted(all_data.keys(), key=mean_y):
        points = np.array(all_data[algo], dtype=object)
        xs = points[:, 2]
        ys = points[:, 3]
        min_x = min([min_x] + [x for x in xs if x > 0])
        max_x = max([max_x] + [x for x in xs if x < 1])
        if len(ys) > 0:
            max_y_data = max(max_y_data, float(np.max(ys)))
        color, faded, linestyle, marker = linestyles[algo]
        (handle,) = plt.plot(
            xs,
            ys,
            "-",
            label=algo,
            color=color,
            ms=7,
            mew=3,
            lw=3,
            marker=marker,
        )
        handles.append(handle)

        labels.append(algo)

    ax = plt.gca()
    y_description = ym["description"]
    if mode == "latency":
        y_description = y_description.replace("(s)", f"({time_unit})")
    ax.set_ylabel(y_description)
    ax.set_xlabel("Recall")
    # Custom scales of the type --x-scale a3
    if x_scale[0] == "a":
        alpha = float(x_scale[1:])

        def fun(x):
            return 1 - (1 - x) ** (1 / alpha)

        def inv_fun(x):
            return 1 - (1 - x) ** alpha

        ax.set_xscale("function", functions=(fun, inv_fun))
        if alpha <= 3:
            ticks = [inv_fun(x) for x in np.arange(0, 1.2, 0.2)]
            plt.xticks(ticks)
        if alpha > 3:
            from matplotlib import ticker

            ax.xaxis.set_major_formatter(ticker.LogitFormatter())
            # plt.xticks(ticker.LogitLocator().tick_values(min_x, max_x))
            plt.xticks([0, 1 / 2, 1 - 1e-1, 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, 1])
    # Other x-scales
    else:
        ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_title(f"{dataset} k={k} batch_size={batch_size}")
    plt.gca().get_position()
    # plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        prop={"size": 9},
    )
    plt.grid(visible=True, which="major", color="0.65", linestyle="-")
    plt.setp(ax.get_xminorticklabels(), visible=True)

    # Logit scale has to be a subset of (0,1)
    if "lim" in xm and x_scale != "logit":
        x0, x1 = xm["lim"]
        # Include data that fall below x_start (e.g. wiki_all_1M with max recall < 0.8)
        left = max(0, min(x0, min_x)) if min_x <= 1 else x0
        right = float(x_max) if x_max is not None else min(x1, 1)
        plt.xlim(left, right)
    elif x_scale == "logit":
        plt.xlim(min_x, max_x)
    if y_max is not None:
        lo = 1e-9 if y_scale == "log" else 0.0
        ax.set_ylim(lo, float(y_max))
    elif "lim" in ym:
        ax.set_ylim(ym["lim"])
    elif mode == "throughput":
        # Throughput (Q/s) can be large; ensure y-axis includes the data (or sensible default)
        lo = 1e-9 if y_scale == "log" else 0.0
        top = (max_y_data * 1.05) if max_y_data > 0 else 1e6
        ax.set_ylim(lo, top)

    # Workaround for bug https://github.com/matplotlib/matplotlib/issues/6789
    ax.spines["bottom"]._adjust_location()

    plt.savefig(fn_out, bbox_inches="tight")
    plt.close()


def create_plot_build(
    build_results, search_results, linestyles, fn_out, dataset, k, batch_size
):
    bt_80 = [0] * len(linestyles)

    bt_90 = [0] * len(linestyles)

    bt_95 = [0] * len(linestyles)

    bt_99 = [0] * len(linestyles)

    data = OrderedDict()
    colors = OrderedDict()

    # Sorting by mean y-value helps aligning plots with labels

    def mean_y(algo):
        points = np.array(search_results[algo], dtype=object)
        return -np.log(np.array(points[:, 3], dtype=np.float32)).mean()

    for pos, algo in enumerate(sorted(search_results.keys(), key=mean_y)):
        points = np.array(search_results[algo], dtype=object)
        # x is recall, ls is algo_name, idxs is index_name
        xs = points[:, 2]
        ls = points[:, 0]
        idxs = points[:, 1]

        len_80, len_90, len_95, len_99 = 0, 0, 0, 0
        for i in range(len(xs)):
            key = (ls[i], idxs[i])
            if key not in build_results:
                continue
            bt = build_results[key][0][2]
            if xs[i] >= 0.80 and xs[i] < 0.90:
                bt_80[pos] = bt_80[pos] + bt
                len_80 = len_80 + 1
            elif xs[i] >= 0.9 and xs[i] < 0.95:
                bt_90[pos] = bt_90[pos] + bt
                len_90 = len_90 + 1
            elif xs[i] >= 0.95 and xs[i] < 0.99:
                bt_95[pos] = bt_95[pos] + bt
                len_95 = len_95 + 1
            elif xs[i] >= 0.99:
                bt_99[pos] = bt_99[pos] + bt
                len_99 = len_99 + 1
        if len_80 > 0:
            bt_80[pos] = bt_80[pos] / len_80
        if len_90 > 0:
            bt_90[pos] = bt_90[pos] / len_90
        if len_95 > 0:
            bt_95[pos] = bt_95[pos] / len_95
        if len_99 > 0:
            bt_99[pos] = bt_99[pos] / len_99
        data[algo] = [
            bt_80[pos],
            bt_90[pos],
            bt_95[pos],
            bt_99[pos],
        ]
        colors[algo] = linestyles[algo][0]

    index = [
        "@80% Recall",
        "@90% Recall",
        "@95% Recall",
        "@99% Recall",
    ]

    df = pd.DataFrame(data, index=index)
    df.replace(0.0, np.nan, inplace=True)
    df = df.dropna(how="all")
    plt.figure(figsize=(12, 9))
    ax = df.plot.bar(rot=0, color=colors)
    fig = ax.get_figure()
    plt.title(
        "Average Build Time within Recall Range "
        f"for k={k} batch_size={batch_size}"
    )
    plt.suptitle(f"{dataset}")
    plt.ylabel("Build Time (s)")
    fig.savefig(fn_out)


def load_lines(results_path, result_files, method, index_key, mode, time_unit):
    results = dict()

    for result_filename in result_files:
        try:
            path = os.path.join(results_path, result_filename)
            if method == "search":
                usecols = ["algo_name", "index_name", "recall", "throughput", "latency"]
            else:
                usecols = ["algo_name", "index_name", "time"]
            try:
                df = pd.read_csv(path, usecols=usecols)
            except ValueError:
                df = pd.read_csv(path)
            df.columns = [str(c).strip().lower() for c in df.columns]
            if df.empty:
                continue
            if method == "search":
                y_col = "throughput" if mode == "throughput" else "latency"
                if y_col not in df.columns or "recall" not in df.columns:
                    continue
            else:
                if "time" not in df.columns:
                    continue

            for _, row in df.iterrows():
                algo_name = str(row["algo_name"]).strip()
                index_name = str(row["index_name"]).strip()
                if index_key == "algo":
                    dict_key = algo_name
                elif index_key == "index":
                    dict_key = (algo_name, index_name)
                if dict_key not in results:
                    results[dict_key] = []
                try:
                    if method == "search":
                        x_val = float(row["recall"])
                        y_val = float(row[y_col])
                        if (
                            mode == "latency"
                            and time_unit != "s"
                        ):
                            y_val = (
                                y_val * (10**3)
                                if time_unit == "ms"
                                else y_val * (10**6)
                            )
                        results[dict_key].append([algo_name, index_name, x_val, y_val])
                    else:
                        val = float(row["time"])
                        results[dict_key].append([algo_name, index_name, val])
                except (ValueError, TypeError, KeyError):
                    continue
        except Exception:
            pass

    return results


def load_overlay_csv(path, label, mode, time_unit):
    """
    Load a single search CSV and return results dict with one key (label).
    Same row format as load_lines: [algo_name, index_name, x_val, y_val].
    """
    out = {label: []}
    usecols = ["algo_name", "index_name", "recall", "throughput", "latency"]
    y_col = "throughput" if mode == "throughput" else "latency"
    try:
        try:
            df = pd.read_csv(path, usecols=usecols)
        except ValueError:
            df = pd.read_csv(path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if df.empty or y_col not in df.columns or "recall" not in df.columns:
            return out
        for _, row in df.iterrows():
            try:
                x_val = float(row["recall"])
                y_val = float(row[y_col])
                if mode == "latency" and time_unit != "s":
                    y_val = (
                        y_val * (10**3) if time_unit == "ms" else y_val * (10**6)
                    )
                index_name = str(row["index_name"]).strip()
                out[label].append([label, index_name, x_val, y_val])
            except (ValueError, TypeError, KeyError):
                continue
    except Exception:
        pass
    return out


def load_all_results(
    dataset_path,
    algorithms,
    groups,
    algo_groups,
    k,
    batch_size,
    method,
    index_key,
    raw,
    mode,
    time_unit,
):
    results_path = os.path.join(dataset_path, "result", method)
    if not os.path.isdir(results_path):
        if method == "build":
            raise FileNotFoundError(
                f"Build result folder not found: {results_path}. "
                "Run the benchmark with --build so that result/build/ exists with CSV files."
            )
        raise FileNotFoundError(f"Result folder not found: {results_path}")
    result_files = os.listdir(results_path)
    if method == "build":
        result_files = [
            result_file
            for result_file in result_files
            if ".csv" in result_file
        ]
    elif method == "search":
        if raw:
            suffix = ",raw"
        else:
            suffix = f",{mode}"
        result_files = [
            result_file
            for result_file in result_files
            if f"{suffix}.csv" in result_file
        ]
    if len(result_files) == 0:
        if method == "build":
            raise FileNotFoundError(
                f"No build CSV files in {results_path}. "
                "Add build CSVs (algo_name, index_name, time) or run benchmark with --build."
            )
        raise FileNotFoundError(f"No CSV result files found in {results_path}")

    if method == "search":
        filter_k_bs = []
        for result_filename in result_files:
            filename_split = result_filename.split(",")
            file_k = file_bs = None
            if len(filename_split) >= 4:
                try:
                    file_k = int(filename_split[-3][1:])
                    file_bs = int(filename_split[-2][2:])
                except (ValueError, IndexError):
                    pass
            else:
                match = re.search(r"_k(\d+)_bs(\d+)(?:,|$)", result_filename)
                if match:
                    file_k = int(match.group(1))
                    file_bs = int(match.group(2))
            if file_k is not None and file_bs is not None and file_k == k and file_bs == batch_size:
                filter_k_bs.append(result_filename)
        result_files = filter_k_bs
        if len(result_files) == 0:
            raise FileNotFoundError(
                f"No search CSV files found for k={k} and batch_size={batch_size} in {results_path}. "
                "Check that the benchmark produced CSVs for these parameters."
            )

    algo_group_files = []
    for result_filename in result_files:
        parts = result_filename.replace(".csv", "").split(",")
        if len(parts) >= 4:
            algo_group_files.append(parts[:2])
        elif len(parts) == 2 and parts[1] in ("latency", "throughput", "raw"):
            # Search CSV named like "cuvs_ivf_pq_base_k10_bs10000,latency.csv"
            base = parts[0]
            segs = base.split("_")
            if "base" in segs and len(segs) > 1:
                idx = segs.index("base")
                algo = "_".join(segs[:idx])
                algo_group_files.append((algo, "base"))
            elif len(segs) >= 2:
                algo_group_files.append((segs[0], segs[1]))
            else:
                algo_group_files.append((base, "base"))
        elif len(parts) == 2:
            algo_group_files.append((parts[0], parts[1]))
        else:
            base = parts[0] if parts else ""
            segs = base.split("_")
            if len(segs) >= 2:
                group = "base" if "base" in segs else segs[1]
                algo_group_files.append((segs[0], group))
            else:
                algo_group_files.append((base, base))
    algo_group_files = list(zip(*algo_group_files)) if algo_group_files else ([], [])

    if len(algorithms) > 0:
        final_results = [
            result_files[i]
            for i in range(len(result_files))
            if (algo_group_files[0][i] in algorithms)
            and (algo_group_files[1][i] in groups)
        ]
    else:
        final_results = [
            result_files[i]
            for i in range(len(result_files))
            if (algo_group_files[1][i] in groups)
        ]

    if len(algo_groups) > 0:
        split_algo_groups = [
            algo_group.split(".") for algo_group in algo_groups
        ]
        split_algo_groups = list(zip(*split_algo_groups))
        final_algo_groups = [
            result_files[i]
            for i in range(len(result_files))
            if (algo_group_files[0][i] in split_algo_groups[0])
            and (algo_group_files[1][i] in split_algo_groups[1])
        ]
        final_results = final_results + final_algo_groups
        final_results = set(final_results)

    results = load_lines(
        results_path, final_results, method, index_key, mode, time_unit
    )
    return results


@click.command()
@click.option("--dataset", default="glove-100-inner", help="Dataset to plot.")
@click.option(
    "--dataset-path",
    default=lambda: os.getenv(
        "RAPIDS_DATASET_ROOT_DIR", os.path.join(os.getcwd(), "datasets/")
    ),
    help="Path to dataset folder.",
)
@click.option(
    "--output-filepath",
    default=os.getcwd(),
    help="Directory where PNG will be saved.",
)
@click.option(
    "--algorithms",
    default=None,
    help="Comma-separated list of named algorithms to plot. If `groups` and "
    "`algo-groups` are both undefined, then group `base` is plotted by "
    "default.",
)
@click.option(
    "--groups",
    default="base",
    help="Comma-separated groups of parameters to plot.",
)
@click.option(
    "--algo-groups",
    help="Comma-separated <algorithm>.<group> to plot. Example usage: "
    '--algo-groups=raft_cagra.large,hnswlib.large".',
)
@click.option(
    "-k",
    "--count",
    default=10,
    type=positive_int,
    help="The number of nearest neighbors to search for.",
)
@click.option(
    "-bs",
    "--batch-size",
    default=10000,
    type=positive_int,
    help="Number of query vectors to use in each query trial.",
)
@click.option("--build", is_flag=True, help="Flag to indicate build mode.")
@click.option("--search", is_flag=True, help="Flag to indicate search mode.")
@click.option(
    "--x-scale",
    default="linear",
    help="Scale to use when drawing the X-axis. Typically linear, "
    "logit, or a2.",
)
@click.option(
    "--y-scale",
    type=click.Choice(
        ["linear", "log", "symlog", "logit"], case_sensitive=False
    ),
    default="linear",
    help="Scale to use when drawing the Y-axis.",
)
@click.option(
    "--x-start",
    default=0.8,
    type=positive_float,
    help="Recall values to start the x-axis from.",
)
@click.option(
    "--x-max",
    default=None,
    type=float,
    help="X-axis upper limit (recall). Optional; if not set, axis uses default (1.0).",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["throughput", "latency"], case_sensitive=False),
    default="latency",
    help="Search mode whose Pareto frontier is used on the Y-axis.",
)
@click.option(
    "--time-unit",
    type=click.Choice(["s", "ms", "us"], case_sensitive=False),
    default="ms",
    help="Time unit to plot when mode is latency.",
)
@click.option(
    "--raw",
    is_flag=True,
    help="Show raw results (not just Pareto frontier) of the mode argument.",
)
@click.option(
    "--y-max",
    default=None,
    type=float,
    help="Y-axis upper limit (e.g. 0.5). Optional; if not set, axis uses metric default or autoscale.",
)
@click.option(
    "--overlay-csv",
    nargs=2,
    type=click.Path(exists=True),
    default=None,
    help="Two search CSV paths to overlay (e.g. normalized and raw IP). Use with --overlay-labels.",
)
@click.option(
    "--overlay-labels",
    nargs=2,
    default=("Normalized", "Raw"),
    help="Labels for the two overlay CSVs (default: Normalized, Raw).",
)
def main(
    dataset: str,
    dataset_path: str,
    output_filepath: str,
    algorithms: str,
    groups: str,
    algo_groups: str,
    count: int,
    batch_size: int,
    build: bool,
    search: bool,
    x_scale: str,
    y_scale: str,
    x_start: float,
    x_max: float,
    mode: str,
    time_unit: str,
    raw: bool,
    y_max: float,
    overlay_csv: tuple,
    overlay_labels: tuple,
) -> None:
    args = locals()

    if args["algorithms"]:
        algorithms = args["algorithms"].split(",")
    else:
        algorithms = []
    groups = args["groups"].split(",")
    if args["algo_groups"]:
        algo_groups = args["algo_groups"].split(",")
    else:
        algo_groups = []
    k = args["count"]
    batch_size = args["batch_size"]
    if not args["build"] and not args["search"]:
        build = True
        search = True
    else:
        build = args["build"]
        search = args["search"]

    search_output_filepath = os.path.join(
        args["output_filepath"],
        f"search-{args['dataset']}-k{k}-batch_size{batch_size}-{args['mode']}.png",
    )
    build_output_filepath = os.path.join(
        args["output_filepath"],
        f"build-{args['dataset']}-k{k}-batch_size{batch_size}.png",
    )

    dataset_dir = os.path.join(args["dataset_path"], args["dataset"])
    build_result_dir = os.path.join(dataset_dir, "result", "build")
    if build and not os.path.isdir(build_result_dir):
        build = False
    elif build:
        build_csvs = [f for f in os.listdir(build_result_dir) if f.endswith(".csv")]
        if not build_csvs:
            build = False

    if args["overlay_csv"] and len(args["overlay_csv"]) == 2:
        search_results = {}
    else:
        search_results = load_all_results(
            os.path.join(args["dataset_path"], args["dataset"]),
            algorithms,
            groups,
            algo_groups,
            k,
            batch_size,
            "search",
            "algo",
            args["raw"],
            args["mode"],
            args["time_unit"],
        )
    if args["overlay_csv"] and len(args["overlay_csv"]) == 2:
        for csv_path, overlay_label in zip(args["overlay_csv"], args["overlay_labels"]):
            extra = load_overlay_csv(
                csv_path,
                overlay_label,
                args["mode"],
                args["time_unit"],
            )
            for key, points in extra.items():
                if points:
                    search_results[key] = points
    linestyles = create_linestyles(sorted(search_results.keys()))
    if search:
        create_plot_search(
            search_results,
            args["x_scale"],
            args["y_scale"],
            search_output_filepath,
            linestyles,
            args["dataset"],
            k,
            batch_size,
            args["mode"],
            args["time_unit"],
            args["x_start"],
            args.get("y_max"),
            args.get("x_max"),
        )
    if build:
        build_results = load_all_results(
            os.path.join(args["dataset_path"], args["dataset"]),
            algorithms,
            groups,
            algo_groups,
            k,
            batch_size,
            "build",
            "index",
            args["raw"],
            args["mode"],
            args["time_unit"],
        )
        create_plot_build(
            build_results,
            search_results,
            linestyles,
            build_output_filepath,
            args["dataset"],
            k,
            batch_size,
        )


if __name__ == "__main__":
    main()
