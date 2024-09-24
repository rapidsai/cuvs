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
import subprocess
import uuid
from typing import Dict, List, Optional, Tuple


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
    raft_log_level: str = "info",
) -> None:
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
    raft_log_level : str, optional
        The logging level for the RAFT library, by default 'info'.

    Returns
    -------
    None
    """
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
            build_file = f"{output_filename[0]}.json"
            temp_build_file = f"{build_file}.lock"
            benchmark_out = os.path.join(build_folder, temp_build_file)
            cmd = [
                ann_executable_path,
                "--build",
                f"--data_prefix={dataset_path}",
                "--benchmark_out_format=json",
                "--benchmark_counters_tabular=true",
                f"--benchmark_out={os.path.join(benchmark_out)}",
                f"--raft_log_level={parse_log_level(raft_log_level)}",
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
                    subprocess.run(cmd, check=True)
                    merge_build_files(
                        build_folder, build_file, temp_build_file
                    )
                except Exception as e:
                    print(f"Error occurred running benchmark: {e}")
                finally:
                    os.remove(os.path.join(build_folder, temp_build_file))
                    if not search:
                        os.remove(temp_conf_filename)

        if search:
            search_folder = os.path.join(legacy_result_folder, "search")
            os.makedirs(search_folder, exist_ok=True)
            search_file = f"{output_filename[1]}.json"
            cmd = [
                ann_executable_path,
                "--search",
                f"--data_prefix={dataset_path}",
                "--benchmark_counters_tabular=true",
                f"--override_kv=k:{k}",
                f"--override_kv=n_queries:{batch_size}",
                "--benchmark_min_warmup_time=1",
                "--benchmark_out_format=json",
                f"--mode={mode}",
                f"--benchmark_out={os.path.join(search_folder, search_file)}",
                f"--raft_log_level={parse_log_level(raft_log_level)}",
            ]
            if force:
                cmd.append("--force")
            if search_threads:
                cmd.append(f"--threads={search_threads}")
            cmd.append(temp_conf_filename)

            if dry_run:
                print(
                    f"Benchmark command for {output_filename[1]}:\n"
                    f"{' '.join(cmd)}\n"
                )
            else:
                try:
                    subprocess.run(cmd, check=True)
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
