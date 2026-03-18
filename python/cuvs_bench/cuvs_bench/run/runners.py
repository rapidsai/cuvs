#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import json
import os
import subprocess
import uuid
import warnings
from typing import Dict, List, Optional, Tuple

from .data_export import (
    clean_algo_name,
    convert_json_to_csv_build,
    convert_json_to_csv_search,
)

def _subprocess_env(ann_executable_path: str) -> Dict[str, str]:
    """Build env for C++ benchmark subprocess. When CUVS_HOME is set, force the repo's libcuvs.so to be used (LD_PRELOAD + LD_LIBRARY_PATH) so the correct local build runs."""
    env = os.environ.copy()
    repo = os.getenv("CUVS_HOME")
    if repo:
        build_dir = os.path.join(repo, "cpp", "build")
        if os.path.isdir(build_dir):
            lib = os.path.join(build_dir, "libcuvs.so")
            if os.path.isfile(lib):
                env["LD_PRELOAD"] = lib + (os.pathsep + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
            env["LD_LIBRARY_PATH"] = build_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            # So IVF-PQ normalization logging goes to a known path (C++ uses this when set)
            log_path = os.path.join(build_dir, "cuvs_ivf_pq_normalization.log")
            env["CUVS_IVF_PQ_NORMALIZATION_LOG"] = log_path
            print(f"[cuvs_bench] IVF-PQ normalization log (if any) -> {log_path}", flush=True)
    return env


def cuvs_bench_cpp(
    conf_file: Dict,
    conf_filename: str,
    conf_filedir: str,
    executables_to_run: Dict[
        Tuple[str, str, Tuple[str, str]], Dict[str, List[Dict]]
    ],
    dataset_path: str,
    force: bool,
    build: bool,
    search: bool,
    dry_run: bool,
    k: int,
    batch_size: int,
    search_threads: Optional[int],
    mode: str = "throughput",
) -> None:
    # So you can confirm which repo's code is running (local vs conda)
    print(f"[cuvs_bench] runners.py loaded from: {os.path.abspath(__file__)}")
    """
    Run the CUVS benchmarking tool with the provided configuration.

    Parameters
    ----------
    conf_file : Dict
        The configuration file content.
    conf_filename : str
        The name of the configuration file.
    conf_filedir : str
        The directory of the configuration file.
    executables_to_run : Dict[Tuple[str, str, Tuple[str, str]],
                         Dict[str, List[Dict]]]
        Dictionary of executables to run and their configurations.
    dataset_path : str
        The path to the dataset.
    force : bool
        Whether to force the execution regardless of existing results.
    build : bool
        Whether to build the indices.
    search : bool
        Whether to perform the search.
    dry_run : bool
        Whether to perform a dry run without actual execution.
    k : int
        The number of nearest neighbors to search for.
    batch_size : int
        The size of each batch for processing.
    search_threads : Optional[int]
        The number of threads to use for searching.
    mode : str, optional
        The mode of search to perform ('latency' or 'throughput'),
        by default 'throughput'.

    Returns
    -------
    None
    """
    warnings.warn(
        "cuvs_bench_cpp() is deprecated and will be removed in a future release. "
        "Use CppGoogleBenchmarkBackend from cuvs_bench.backends instead.",
        FutureWarning,
        stacklevel=2,
    )

    for (
        executable,
        ann_executable_path,
        output_filename,
    ) in executables_to_run.keys():
        # Need to write temporary configuration
        temp_conf_filename = (
            f"{conf_filename}_{output_filename[1]}_{uuid.uuid1()}.json"
        )
        with open(temp_conf_filename, "w") as f:
            temp_conf = {
                "dataset": conf_file["dataset"],
                "search_basic_param": conf_file["search_basic_param"],
                "index": executables_to_run[
                    (executable, ann_executable_path, output_filename)
                ]["index"],
            }
            json_str = json.dumps(temp_conf, indent=2)
            f.write(json_str)

        legacy_result_folder = os.path.join(
            dataset_path, conf_file["dataset"]["name"], "result"
        )
        os.makedirs(legacy_result_folder, exist_ok=True)

        if build:
            build_folder = os.path.join(legacy_result_folder, "build")
            os.makedirs(build_folder, exist_ok=True)
            # Use underscores so --benchmark_out is not truncated by comma
            build_file = f"{output_filename[0].replace(',', '_')}.json"
            temp_build_file = f"{build_file}.lock"
            benchmark_out = os.path.join(build_folder, temp_build_file)
            cmd = [
                ann_executable_path,
                "--build",
                f"--data_prefix={dataset_path}",
                "--benchmark_out_format=json",
                "--benchmark_counters_tabular=true",
                f"--benchmark_out={os.path.join(benchmark_out)}",
            ]
            if force:
                cmd.append("--force")
            cmd.append(temp_conf_filename)

            if dry_run:
                print(
                    f"Benchmark command for {output_filename[0]}:\n"
                    f"{' '.join(cmd)}\n"
                )
            else:
                try:
                    subprocess.run(cmd, check=True, env=_subprocess_env(ann_executable_path))
                    merge_build_files(
                        build_folder, build_file, temp_build_file
                    )
                    # Update build CSV after each build so CSVs are written incrementally
                    _dataset = conf_file["dataset"]["name"]
                    print(f"[cuvs_bench] Converting build JSON -> CSV (dataset={_dataset}, path={os.path.abspath(dataset_path)})")
                    convert_json_to_csv_build(_dataset, dataset_path)
                except Exception as e:
                    print(f"Error occurred running benchmark: {e}")
                finally:
                    os.remove(os.path.join(build_folder, temp_build_file))
                    if not search:
                        os.remove(temp_conf_filename)

        if search:
            search_folder = os.path.join(legacy_result_folder, "search")
            os.makedirs(search_folder, exist_ok=True)
            # Use underscores in filename so --benchmark_out is not truncated by comma
            search_file_safe = f"{output_filename[1].replace(',', '_')}.json"
            final_search_json = os.path.join(search_folder, search_file_safe)
            cmd_base = [
                ann_executable_path,
                "--search",
                f"--data_prefix={dataset_path}",
                "--benchmark_counters_tabular=true",
                f"--override_kv=k:{k}",
                f"--override_kv=n_queries:{batch_size}",
                "--benchmark_min_warmup_time=4",
                "--benchmark_out_format=json",
                f"--mode={mode}",
            ]
            if force:
                cmd_base.append("--force")
            if search_threads:
                cmd_base.append(f"--threads={search_threads}")
            # C++ binary expects the config file as the *last* argument (argv[--argc])
            # So --benchmark_out must come before the config file.
            cmd_base.append(temp_conf_filename)

            if dry_run:
                cmd = cmd_base[:-1] + [f"--benchmark_out={final_search_json}", temp_conf_filename]
                print(
                    f"Benchmark command for {output_filename[1]}:\n"
                    f"{' '.join(cmd)}\n"
                )
            else:
                cmd = cmd_base[:-1] + [f"--benchmark_out={final_search_json}", temp_conf_filename]
                try:
                    subprocess.run(cmd, check=True, env=_subprocess_env(ann_executable_path))
                    _dataset = conf_file["dataset"]["name"]
                    convert_json_to_csv_search(_dataset, dataset_path)
                except Exception as e:
                    print(f"Error occurred running benchmark: {e}")
                finally:
                    os.remove(temp_conf_filename)


log_levels = {
    "off": 0,
    "error": 1,
    "warn": 2,
    "info": 3,
    "debug": 4,
    "trace": 5,
}


def parse_log_level(level_str: str) -> int:
    """
    Parse the log level from string to integer.

    Parameters
    ----------
    level_str : str
        The log level as a string.

    Returns
    -------
    int
        The corresponding integer value of the log level.

    Raises
    ------
    ValueError
        If the log level string is invalid.
    """
    if level_str not in log_levels:
        raise ValueError(f"Invalid log level: {level_str}")
    return log_levels[level_str.lower()]


def merge_build_files(
    build_dir: str, build_file: str, temp_build_file: str
) -> None:
    """
    Merge temporary build files into the main build file.

    Parameters
    ----------
    build_dir : str
        The directory of the build files.
    build_file : str
        The main build file.
    temp_build_file : str
        The temporary build file to merge.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the temporary build file is not found.
    """
    build_dict = {}

    # If build file exists, read it
    build_json_path = os.path.join(build_dir, build_file)
    tmp_build_json_path = os.path.join(build_dir, temp_build_file)
    if os.path.isfile(build_json_path):
        try:
            with open(build_json_path, "r") as f:
                build_dict = json.load(f)
        except Exception as e:
            print(
                f"Error loading existing build file: {build_json_path} ({e})"
            )

    temp_build_dict = {}
    if os.path.isfile(tmp_build_json_path):
        with open(tmp_build_json_path, "r") as f:
            temp_build_dict = json.load(f)
    else:
        raise ValueError(f"Temp build file not found: {tmp_build_json_path}")

    tmp_benchmarks = temp_build_dict.get("benchmarks", {})
    benchmarks = build_dict.get("benchmarks", {})

    # If the build time is absolute 0 then an error occurred
    final_bench_dict = {b["name"]: b for b in benchmarks if b["real_time"] > 0}

    for tmp_bench in tmp_benchmarks:
        if tmp_bench["real_time"] > 0:
            final_bench_dict[tmp_bench["name"]] = tmp_bench

    temp_build_dict["benchmarks"] = list(final_bench_dict.values())
    with open(build_json_path, "w") as f:
        json_str = json.dumps(temp_build_dict, indent=2)
        f.write(json_str)
