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

import importlib
import itertools
import json
import os
import subprocess
import sys
import uuid
import warnings
import yaml

from importlib import import_module
from typing import Optional, Tuple, Union, Dict, Any

from .runners import cuvs_bench_cpp


def run_benchmark(
    subset_size: int,
    count: int,
    batch_size: int,
    dataset_configuration: Optional[str],
    configuration: Optional[str],
    dataset: str,
    dataset_path: str,
    build: Optional[bool],
    search: Optional[bool],
    algorithms: Optional[str],
    groups: str,
    algo_groups: Optional[str],
    force: bool,
    search_mode: str,
    search_threads: int,
    dry_run: bool,
    raft_log_level: int
) -> None:
    """
    Runs a benchmarking process based on the provided configurations.

    Parameters
    ----------
    count : int
        The number of iterations to run.
    batch_size : int
        The size of each batch for processing.
    dataset_configuration : Optional[str]
        Path to the dataset configuration file.
    configuration : Optional[str]
        Path to the algorithm configuration file or directory.
    dataset : str
        The name of the dataset to use.
    dataset_path : str
        The path to the dataset directory.
    build : Optional[bool]
        Whether to build the indices.
    search : Optional[bool]
        Whether to perform the search.
    algorithms : Optional[str]
        Comma-separated list of algorithm names to use.
    groups : str
        Comma-separated list of groups to consider.
    algo_groups : Optional[str]
        Comma-separated list of algorithm groups to consider.
    force : bool
        Whether to force the execution regardless of warnings.
    search_mode : str
        The mode of search to perform.
    search_threads : int
        The number of threads to use for searching.
    dry_run : bool
        Whether to perform a dry run without actual execution.
    raft_log_level : int
        The logging level for the RAFT library.

    Returns
    -------
    None
    """

    scripts_path = os.path.dirname(os.path.realpath(__file__))
    call_path = os.getcwd()
    gpu_present = rmm_present()

    with open(os.path.join(scripts_path, "../config", "algorithms.yaml"), "r") as f:
        algos_yaml = yaml.safe_load(f)

    # If both build and search are not provided,
    # run both
    if not build and not search:
        build = True
        search = True

    # look for dataset configuration path, if not given then use the
    # default location in cuvs_bench/conf
    if dataset_configuration:
        dataset_conf_f = dataset_configuration
    else:
        dataset_conf_f = os.path.join(scripts_path, "../config/datasets", "datasets.yaml")
    with open(dataset_conf_f, "r") as f:
        dataset_conf_all = yaml.safe_load(f)

    # load datasets configuration files
    dataset_conf = None
    for dset in dataset_conf_all:
        if dataset == dset["name"]:
            dataset_conf = dset
            break
    if not dataset_conf:
        raise ValueError("Could not find a dataset configuration")

    conf_file = dict()
    conf_file["dataset"] = dataset_conf
    if subset_size:
        conf_file["dataset"]["subset_size"] = subset_size

    conf_file["search_basic_param"] = {}
    conf_file["search_basic_param"]["k"] = count
    conf_file["search_basic_param"]["batch_size"] = batch_size

    algos_conf_fs = os.listdir(os.path.join(scripts_path, "../config", "algos"))
    algos_conf_fs = [
        os.path.join(scripts_path, "../config", "algos", f)
        for f in algos_conf_fs
        if ".json" not in f and "constraint" not in f and ".py" not in f
    ]
    conf_filedir = os.path.join(scripts_path, "conf", "algos")
    if configuration:
        if os.path.isdir(configuration):
            conf_filedir = configuration
            algos_conf_fs = algos_conf_fs + [
                os.path.join(configuration, f)
                for f in os.listdir(configuration)
                if ".json" not in f
            ]
        elif os.path.isfile(configuration):
            conf_filedir = os.path.normpath(configuration).split(os.sep)
            conf_filedir = os.path.join(*conf_filedir[:-1])
            algos_conf_fs = algos_conf_fs + [configuration]

    filter_algos = True if algorithms else False
    if filter_algos:
        allowed_algos = algorithms.split(",")
    named_groups = groups.split(",")
    filter_algo_groups = True if algo_groups else False
    allowed_algo_groups = None
    if filter_algo_groups:
        allowed_algo_groups = [
            algo_group.split(".") for algo_group in algo_groups.split(",")
        ]
        allowed_algo_groups = list(zip(*allowed_algo_groups))
    algos_conf = dict()
    for algo_f in algos_conf_fs:
        with open(algo_f, "r") as f:
            try:
                algo = yaml.safe_load(f)
            except Exception as e:
                warnings.warn(
                    f"Could not load YAML config {algo_f} due to "
                    + e.with_traceback()
                )
                continue
            insert_algo = True
            insert_algo_group = False
            if filter_algos:
                if algo["name"] not in allowed_algos:
                    insert_algo = False
            if filter_algo_groups:
                if algo["name"] in allowed_algo_groups[0]:
                    insert_algo_group = True

            def add_algo_group(group_list):
                if algo["name"] not in algos_conf:
                    algos_conf[algo["name"]] = {"groups": {}}
                for group in algo["groups"].keys():
                    if group in group_list:
                        algos_conf[algo["name"]]["groups"][group] = algo[
                            "groups"
                        ][group]
                if "constraints" in algo:
                    algos_conf[algo["name"]]["constraints"] = algo[
                        "constraints"
                    ]

            if insert_algo:
                add_algo_group(named_groups)
            if insert_algo_group:
                add_algo_group(allowed_algo_groups[1])

    executables_to_run = dict()
    for algo in algos_conf.keys():
        validate_algorithm(algos_yaml, algo, gpu_present)
        for group in algos_conf[algo]["groups"].keys():
            executable = find_executable(
                algos_yaml, algo, group, count, batch_size
            )
            if executable not in executables_to_run:
                executables_to_run[executable] = {"index": []}
            build_params = algos_conf[algo]["groups"][group]["build"] or {}
            search_params = algos_conf[algo]["groups"][group]["search"] or {}

            param_names = []
            param_lists = []
            for param in build_params.keys():
                param_names.append(param)
                param_lists.append(build_params[param])

            all_build_params = itertools.product(*param_lists)

            search_param_names = []
            search_param_lists = []
            for search_param in search_params.keys():
                search_param_names.append(search_param)
                search_param_lists.append(search_params[search_param])

            for params in all_build_params:
                index = {"algo": algo, "build_param": {}}
                if group != "base":
                    index_name = f"{algo}_{group}"
                else:
                    index_name = f"{algo}"
                for i in range(len(params)):
                    index["build_param"][param_names[i]] = params[i]
                    index_name += "." + f"{param_names[i]}{params[i]}"

                if "constraints" in algos_conf[algo]:
                    if "build" in algos_conf[algo]["constraints"]:
                        importable = algos_conf[algo]["constraints"]["build"]
                        importable = importable.split(".")
                        module = ".".join(importable[:-1])
                        func = importable[-1]
                        validator = import_module(module)
                        build_constraints = getattr(validator, func)
                        if "dims" not in conf_file["dataset"]:
                            raise ValueError(
                                "`dims` needed for build constraints but not "
                                "specified in datasets.yaml"
                            )
                        if not build_constraints(
                            index["build_param"], conf_file["dataset"]["dims"]
                        ):
                            continue
                index_filename = (
                    index_name
                    if len(index_name) < 128
                    else str(hash(index_name))
                )
                index["name"] = index_name
                index["file"] = os.path.join(
                    dataset_path, dataset, "index", index_filename
                )
                index["search_params"] = []
                all_search_params = itertools.product(*search_param_lists)
                for search_params in all_search_params:
                    search_dict = dict()
                    for i in range(len(search_params)):
                        search_dict[search_param_names[i]] = search_params[i]
                    if "constraints" in algos_conf[algo]:
                        if "search" in algos_conf[algo]["constraints"]:
                            importable = algos_conf[algo]["constraints"][
                                "search"
                            ]
                            importable = importable.split(".")
                            module = ".".join(importable[:-1])
                            func = importable[-1]
                            validator = import_module(module)
                            search_constraints = getattr(validator, func)
                            if search_constraints(
                                search_dict,
                                index["build_param"],
                                count,
                                batch_size,
                            ):
                                index["search_params"].append(search_dict)
                    else:
                        index["search_params"].append(search_dict)
                executables_to_run[executable]["index"].append(index)

                if len(index["search_params"]) == 0:
                    print("No search parameters were added to configuration")
            executable = find_executable(
                algos_yaml, algo, group, count, batch_size
            )
            if executable not in executables_to_run:
                executables_to_run[executable] = {"index": []}
            build_params = algos_conf[algo]["groups"][group]["build"] or {}
            search_params = algos_conf[algo]["groups"][group]["search"] or {}

            param_names = []
            param_lists = []
            for param in build_params.keys():
                param_names.append(param)
                param_lists.append(build_params[param])

            all_build_params = itertools.product(*param_lists)

            search_param_names = []
            search_param_lists = []
            for search_param in search_params.keys():
                search_param_names.append(search_param)
                search_param_lists.append(search_params[search_param])

            for params in all_build_params:
                index = {"algo": algo, "build_param": {}}
                if group != "base":
                    index_name = f"{algo}_{group}"
                else:
                    index_name = f"{algo}"
                for i in range(len(params)):
                    index["build_param"][param_names[i]] = params[i]
                    index_name += "." + f"{param_names[i]}{params[i]}"

                if "constraints" in algos_conf[algo]:
                    if "build" in algos_conf[algo]["constraints"]:
                        importable = algos_conf[algo]["constraints"]["build"]
                        importable = importable.split(".")
                        module = ".".join(importable[:-1])
                        func = importable[-1]
                        validator = import_module(module)
                        build_constraints = getattr(validator, func)
                        if "dims" not in conf_file["dataset"]:
                            raise ValueError(
                                "`dims` needed for build constraints but not "
                                "specified in datasets.yaml"
                            )
                        if not build_constraints(
                            index["build_param"], conf_file["dataset"]["dims"]
                        ):
                            continue
                index_filename = (
                    index_name
                    if len(index_name) < 128
                    else str(hash(index_name))
                )
                index["name"] = index_name
                index["file"] = os.path.join(
                    dataset_path, dataset, "index", index_filename
                )
                index["search_params"] = []
                all_search_params = itertools.product(*search_param_lists)
                for search_params in all_search_params:
                    search_dict = dict()
                    for i in range(len(search_params)):
                        search_dict[search_param_names[i]] = search_params[i]
                    # if "constraints" in algos_conf[algo]:
                    # todo: refactor common code
                    if False:
                        if "search" in algos_conf[algo]["constraints"]:
                            if validate_constraints(algos_conf,
                                                    algo,
                                                    "search",
                                                    search_dict,
                                                    index["build_param"],
                                                    count,
                                                    batch_size):
                                index["search_params"].append(search_dict)
                        else:
                            index["search_params"].append(search_dict)
                executables_to_run[executable]["index"].append(index)

                if len(index["search_params"]) == 0:
                    print("No search parameters were added to configuration")

    cuvs_bench_cpp(
        conf_file,
        f"{dataset}",
        conf_filedir,
        executables_to_run,
        dataset_path,
        force,
        build,
        search,
        dry_run,
        count,
        batch_size,
        search_threads,
        search_mode,
        raft_log_level,
    )


def rmm_present() -> bool:
    """
    Check if RMM is present.

    Returns
    -------
    bool
        True if RMM is present, False otherwise.
    """
    try:
        import rmm  # noqa: F401
        return True
    except ImportError:
        return False


def find_executable(algos_conf: dict, algo: str, group: str, k: int, batch_size: int) -> Tuple[str, str, Tuple[str, str]]:
    """
    Find the executable for the given algorithm and group.

    Parameters
    ----------
    algos_conf : dict
        The configuration dictionary for the algorithms.
    algo : str
        The name of the algorithm.
    group : str
        The name of the group.
    k : int
        The number of nearest neighbors to search for.
    batch_size : int
        The size of each batch for processing.

    Returns
    -------
    Tuple[str, str, Tuple[str, str]]
        A tuple containing the executable name, the path to the executable, and the file name.
    """
    executable = algos_conf[algo]["executable"]
    file_name = (f"{algo},{group}", f"{algo},{group},k{k},bs{batch_size}")

    # Check for devcontainer build
    scripts_path = os.path.dirname(os.path.realpath(__file__))
    build_path = "/home/coder/cuvs/cpp/build/latest/bench/ann"
    print(f"build_path: {build_path}")
    if os.path.exists(build_path):
        print(f"-- Detected devcontainer artifacts in {build_path}. ")
        return executable, build_path, file_name

    build_path = os.getenv("CUVS_HOME")
    if build_path is not None:
        build_path = os.path.join(build_path, "cpp", "build", "release", executable)
        if os.path.exists(build_path):
            print(f"-- Using RAFT bench from repository in {build_path}. ")
            return executable, build_path, file_name

    # # todo: better path detection for devcontainer
    # build_path = os.getenv("CUVS_BENCH_BUILD_PATH")
    # print("build_path: ", build_path)
    # if build_path is not None:
    #     if os.path.exists(build_path):
    #         print(f"-- Using devcontainer location from {build_path}. ")
            # return executable, build_path, file_name

    conda_path = os.getenv("CONDA_PREFIX")
    if conda_path is not None:
        conda_path = os.path.join(conda_path, "bin", "ann", executable)
        if os.path.exists(conda_path):
            print("-- Using cuVS bench found in conda environment. ")
            return executable, conda_path, file_name
        else:
            raise FileNotFoundError(executable)
    else:
        raise FileNotFoundError(executable)


def validate_algorithm(algos_conf: dict, algo: str, gpu_present: bool) -> bool:
    """
    Validate algorithm and whether it requires gpu. .

    Parameters
    ----------
    algos_conf : dict
        The configuration dictionary for the algorithms.
    algo : str
        The name of the algorithm.
    gpu_present : bool
        Whether a GPU is present.

    Returns
    -------
    bool
        True if the algorithm is valid for the current hardware configuration, False otherwise.
    """
    algos_conf_keys = set(algos_conf.keys())
    if gpu_present:
        return algo in algos_conf_keys
    else:
        return algo in algos_conf_keys and algos_conf[algo]["requires_gpu"] is False


def validate_constraints(
    algos_conf: Dict[str, Any],
    algo: str,
    constraint_type: str,
    param: Dict[str, Any],
    dims: Any,
    k: Optional[int],
    batch_size: Optional[int]
) -> bool:
    """
    Validate the constraints for the given algorithm and constraint type.

    Parameters
    ----------
    algos_conf : Dict[str, Any]
        The configuration dictionary for the algorithms.
    algo : str
        The name of the algorithm.
    constraint_type : str
        The type of constraint to validate ('build' or 'search').
    param : Dict[str, Any]
        The parameters to validate against the constraints.
    dims : Any
        The dimensions required for the constraints.

    Returns
    -------
    bool
        True if the constraints are valid, False otherwise.
    """
    if constraint_type in algos_conf[algo]["constraints"]:
        importable = algos_conf[algo]["constraints"][constraint_type]
        importable = importable.split(".")
        module = ".".join(importable[:-1])
        func = importable[-1]
        print(f"module: {module}")
        validator = importlib.import_module(module)
        constraints_func = getattr(validator, func)
        if constraint_type == "build" and "dims" not in conf_file["dataset"]:
            raise ValueError("`dims` needed for build constraints but not specified in datasets.yaml")
        return constraints_func(param, dims)
    return True
