#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""
Config Loaders for cuvs-bench orchestrator.

This module defines the abstract ConfigLoader interface and backend-specific
implementations that handle configuration loading and preprocessing.
"""

import itertools
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class IndexConfig:
    """
    Configuration for a single index (one build configuration).

    Attributes
    ----------
    name : str
        Index name (e.g., 'cuvs_cagra.graph_degree32.intermediate_graph_degree32')
    algo : str
        Algorithm name (e.g., 'cuvs_cagra')
    build_param : Dict[str, Any]
        Build parameters (e.g., {"graph_degree": 32})
    search_params : List[Dict[str, Any]]
        List of search parameter combinations for this index
    file : str
        Path where index will be stored
    """

    name: str
    algo: str
    build_param: Dict[str, Any]
    search_params: List[Dict[str, Any]]
    file: str


@dataclass
class BenchmarkConfig:
    """
    Configuration for a benchmark run (may contain multiple indexes).

    This is the standardized format that all ConfigLoaders produce,
    regardless of the backend type. For C++ backend, multiple indexes
    are batched into ONE config and ONE C++ command (matching runners.py).

    Attributes
    ----------
    indexes : List[IndexConfig]
        List of index configurations to run together
    backend_config : Dict[str, Any]
        Backend-specific configuration (e.g., executable_path for C++,
        host/port for Milvus)
    """

    indexes: List[IndexConfig]
    backend_config: Dict[str, Any] = field(default_factory=dict)

    # Convenience properties for single-index access (backward compatibility)
    @property
    def index_name(self) -> str:
        return self.indexes[0].name if self.indexes else ""

    @property
    def algo(self) -> str:
        return self.indexes[0].algo if self.indexes else ""

    @property
    def build_params(self) -> Dict[str, Any]:
        return self.indexes[0].build_param if self.indexes else {}

    @property
    def search_params_list(self) -> List[Dict[str, Any]]:
        return self.indexes[0].search_params if self.indexes else []

    @property
    def index_path(self) -> Path:
        return Path(self.indexes[0].file) if self.indexes else Path("")


@dataclass
class DatasetConfig:
    """
    Dataset configuration for benchmarking.

    Attributes
    ----------
    name : str
        Dataset name
    base_file : Optional[str]
        Path to base vectors file
    query_file : Optional[str]
        Path to query vectors file
    groundtruth_neighbors_file : Optional[str]
        Path to ground truth file
    distance : str
        Distance metric
    dims : Optional[int]
        Vector dimensions
    subset_size : Optional[int]
        Subset size for testing
    """

    name: str
    base_file: Optional[str] = None
    query_file: Optional[str] = None
    groundtruth_neighbors_file: Optional[str] = None
    distance: str = "euclidean"
    dims: Optional[int] = None
    subset_size: Optional[int] = None


class ConfigLoader(ABC):
    """
    Abstract base class for configuration loaders.

    Each backend type has its own ConfigLoader that knows how to:
    - Load configuration from files or other sources
    - Expand parameter combinations
    - Validate constraints
    - Produce standardized BenchmarkConfig objects
    """

    @abstractmethod
    def load(self, **kwargs) -> Tuple[DatasetConfig, List[BenchmarkConfig]]:
        """
        Load and prepare benchmark configurations.

        Returns
        -------
        Tuple[DatasetConfig, List[BenchmarkConfig]]
            Dataset configuration and list of benchmark configurations to run
        """
        pass

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Return the backend type this loader is for (e.g., 'cpp_gbench')."""
        pass


class CppGBenchConfigLoader(ConfigLoader):
    """
    Configuration loader for C++ Google Benchmark backend.

    This loader handles:
    - Loading YAML configuration files
    - Finding C++ executables
    - Expanding build/search parameter combinations
    - Validating constraints

    All the C++ specific preprocessing logic from run.py is encapsulated here.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config loader.

        Parameters
        ----------
        config_path : Optional[str]
            Path to config directory. If None, uses default path.
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../config"
        )
        self._gpu_present: Optional[bool] = None

    @property
    def backend_type(self) -> str:
        return "cpp_gbench"

    @property
    def gpu_present(self) -> bool:
        """Check if GPU is available (cached)."""
        if self._gpu_present is None:
            try:
                import rmm  # noqa: F401

                self._gpu_present = True
            except ImportError:
                self._gpu_present = False
        return self._gpu_present

    def load(
        self,
        dataset: str,
        dataset_path: str,
        dataset_configuration: Optional[str] = None,
        algorithm_configuration: Optional[str] = None,
        algorithms: Optional[str] = None,
        groups: Optional[str] = None,
        algo_groups: Optional[str] = None,
        count: int = 10,
        batch_size: int = 10000,
        subset_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[DatasetConfig, List[BenchmarkConfig]]:
        """
        Load C++ benchmark configurations from YAML files.

        Parameters
        ----------
        dataset : str
            Dataset name
        dataset_path : str
            Path to dataset directory
        dataset_configuration : Optional[str]
            Path to dataset configuration file
        algorithm_configuration : Optional[str]
            Path to algorithm configuration directory or file
        algorithms : Optional[str]
            Comma-separated list of algorithms to run
        groups : Optional[str]
            Comma-separated list of groups to run
        algo_groups : Optional[str]
            Comma-separated list of algorithm groups
        count : int
            Number of neighbors (k)
        batch_size : int
            Batch size for search
        subset_size : Optional[int]
            Dataset subset size
        **kwargs
            Additional keyword arguments. In tune mode, the orchestrator
            passes these internal kwargs:

            - _tune_mode : bool
                If True, returns single config with exact params instead of
                Cartesian product expansion from YAML.
            - _tune_build_params : dict
                Exact build parameters suggested by Optuna
                (e.g., {"nlist": 5347})
            - _tune_search_params : dict
                Exact search parameters suggested by Optuna
                (e.g., {"nprobe": 73})

        Returns
        -------
        Tuple[DatasetConfig, List[BenchmarkConfig]]
            Dataset config and list of benchmark configs.
            - Sweep mode: Multiple configs (Cartesian product of YAML params)
            - Tune mode: Single config (exact Optuna-suggested params)
        """
        # Extract tune mode kwargs (passed by orchestrator._run_trial)
        tune_mode = kwargs.pop("_tune_mode", False)
        tune_build_params = kwargs.pop("_tune_build_params", None)
        tune_search_params = kwargs.pop("_tune_search_params", None)

        config_path = self.config_path

        # Load dataset configuration
        dataset_conf_all = self.load_yaml_file(
            dataset_configuration
            or os.path.join(config_path, "datasets", "datasets.yaml")
        )
        dataset_conf = self.get_dataset_configuration(
            dataset, dataset_conf_all
        )

        # Create DatasetConfig
        dataset_config = DatasetConfig(
            name=dataset_conf["name"],
            base_file=dataset_conf.get("base_file"),
            query_file=dataset_conf.get("query_file"),
            groundtruth_neighbors_file=dataset_conf.get(
                "groundtruth_neighbors_file"
            ),
            distance=dataset_conf.get("distance", "euclidean"),
            dims=dataset_conf.get("dims"),
            subset_size=subset_size,
        )

        # Prepare conf_file for constraint validation
        conf_file = {"dataset": dataset_conf}
        if subset_size:
            conf_file["dataset"]["subset_size"] = subset_size
        conf_file["search_basic_param"] = {
            "k": count,
            "batch_size": batch_size,
        }

        # Gather and load algorithm configs
        algos_conf_fs = self.gather_algorithm_configs(
            config_path, algorithm_configuration
        )

        allowed_algos = algorithms.split(",") if algorithms else None
        allowed_groups = groups.split(",") if groups else None
        allowed_algo_groups = (
            [algo_group.split(".") for algo_group in algo_groups.split(",")]
            if algo_groups
            else None
        )

        algos_conf = self.load_algorithms_conf(
            algos_conf_fs,
            allowed_algos,
            allowed_groups,
            tuple(zip(*allowed_algo_groups)) if allowed_algo_groups else None,
        )

        # Load algorithms.yaml for executable info
        algos_yaml = self.load_yaml_file(
            os.path.join(config_path, "algorithms.yaml")
        )

        # Prepare executables and convert to BenchmarkConfig list
        benchmark_configs = self._prepare_benchmark_configs(
            algos_conf,
            algos_yaml,
            self.gpu_present,
            conf_file,
            dataset_path,
            dataset,
            count,
            batch_size,
            tune_mode=tune_mode,
            tune_build_params=tune_build_params,
            tune_search_params=tune_search_params,
        )

        return dataset_config, benchmark_configs

    def _prepare_benchmark_configs(
        self,
        algos_conf: dict,
        algos_yaml: dict,
        gpu_present: bool,
        conf_file: dict,
        dataset_path: str,
        dataset: str,
        count: int,
        batch_size: int,
        tune_mode: bool = False,
        tune_build_params: dict = None,
        tune_search_params: dict = None,
    ) -> List[BenchmarkConfig]:
        """
        Prepare list of BenchmarkConfig from algorithm configurations.

        This matches the old workflow (run.py prepare_executables) by grouping
        ALL indexes for the same executable into ONE BenchmarkConfig.
        This ensures ONE C++ command runs ALL indexes together.

        In tune mode, generates a single config with exact Optuna-suggested params
        instead of Cartesian product expansion.
        """
        # Group indexes by executable (matches run.py executables_to_run structure)
        # Key: (executable, executable_path, output_filename)
        # Value: {"index": [...], "algo_info": {...}}
        executables_to_run = {}

        for algo, algo_conf in algos_conf.items():
            if not self.validate_algorithm(algos_yaml, algo, gpu_present):
                continue

            for group, group_conf in algo_conf["groups"].items():
                try:
                    executable, executable_path, file_name = (
                        self.find_executable(
                            algos_yaml, algo, group, count, batch_size
                        )
                    )
                except FileNotFoundError:
                    warnings.warn(f"Executable not found for {algo}")
                    continue

                # Get algorithm info from algorithms.yaml
                algo_info = algos_yaml.get(algo, {})

                # Prepare index configurations
                indexes = self.prepare_indexes(
                    group_conf,
                    algo,
                    group,
                    conf_file,
                    algos_conf,
                    dataset_path,
                    dataset,
                    count,
                    batch_size,
                    tune_mode=tune_mode,
                    tune_build_params=tune_build_params,
                    tune_search_params=tune_search_params,
                )

                # Group by executable (matches run.py)
                key = (executable, executable_path, file_name)
                if key not in executables_to_run:
                    executables_to_run[key] = {
                        "index": [],
                        "algo_info": algo_info,
                        "dataset_path": dataset_path,
                        "dataset": dataset,
                    }
                executables_to_run[key]["index"].extend(indexes)

        # Convert grouped indexes to BenchmarkConfig objects
        # ONE BenchmarkConfig per executable (with ALL indexes)
        benchmark_configs = []
        for (
            executable,
            executable_path,
            file_name,
        ), data in executables_to_run.items():
            # Convert raw index dicts to IndexConfig objects
            index_configs = [
                IndexConfig(
                    name=idx["name"],
                    algo=idx["algo"],
                    build_param=idx["build_param"],
                    search_params=idx.get("search_params", [{}]),
                    file=idx["file"],
                )
                for idx in data["index"]
            ]

            config = BenchmarkConfig(
                indexes=index_configs,
                backend_config={
                    "executable_path": executable_path,
                    "requires_gpu": data["algo_info"].get(
                        "requires_gpu", False
                    ),
                    "data_prefix": data["dataset_path"],
                    "dataset": data["dataset"],
                    "output_filename": file_name,  # Tuple: (build_name, search_name)
                    "algo": index_configs[0].algo if index_configs else "",
                },
            )
            benchmark_configs.append(config)

        return benchmark_configs

    # =========================================================================
    # `Helper methods (copied from run.py as-is)`
    # =========================================================================

    def load_yaml_file(self, file_path: str) -> dict:
        """
        Load a YAML file and return its contents as a dictionary.

        Parameters
        ----------
        file_path : str
            The path to the YAML file.

        Returns
        -------
        dict
            The contents of the YAML file.
        """
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    def get_dataset_configuration(
        self, dataset: str, dataset_conf_all: list
    ) -> dict:
        """
        Retrieve the configuration for a specific dataset.

        Parameters
        ----------
        dataset : str
            The name of the dataset to retrieve the configuration for.
        dataset_conf_all : list
            A list of dataset configurations.

        Returns
        -------
        dict
            The configuration for the specified dataset.

        Raises
        ------
        ValueError
            If the dataset configuration is not found.
        """
        for d in dataset_conf_all:
            if dataset == d["name"]:
                return d
        raise ValueError("Could not find a dataset configuration")

    def gather_algorithm_configs(
        self, config_path: str, algorithm_configuration: Optional[str]
    ) -> list:
        """
        Gather the list of algorithm configuration files.

        Parameters
        ----------
        config_path : str
            The path to the config directory.
        algorithm_configuration : Optional[str]
            The path to the algorithm configuration directory or file.

        Returns
        -------
        list
            A list of paths to the algorithm configuration files.
        """
        algos_conf_fs = os.listdir(os.path.join(config_path, "algos"))
        algos_conf_fs = [
            os.path.join(config_path, "algos", f)
            for f in algos_conf_fs
            if ".json" not in f
            and "constraint" not in f
            and ".py" not in f
            and "__pycache__" not in f
        ]

        if algorithm_configuration:
            if os.path.isdir(algorithm_configuration):
                algos_conf_fs += [
                    os.path.join(algorithm_configuration, f)
                    for f in os.listdir(algorithm_configuration)
                    if ".json" not in f
                ]
            elif os.path.isfile(algorithm_configuration):
                algos_conf_fs.append(algorithm_configuration)
        return algos_conf_fs

    def load_algorithms_conf(
        self,
        algos_conf_fs: list,
        allowed_algos: Optional[list],
        allowed_groups: Optional[list],
        allowed_algo_groups: Optional[tuple],
    ) -> dict:
        """
        Load and filter the algorithm configurations.

        Parameters
        ----------
        algos_conf_fs : list
            A list of paths to algorithm configuration files.
        allowed_algos : Optional[list]
            A list of allowed algorithm names to filter by.
        allowed_algo_groups : Optional[tuple]
            A tuple of allowed algorithm groups to filter by.

        Returns
        -------
        dict
            A dictionary containing the loaded and filtered algorithm
            configurations.
        """
        algos_conf = {}
        for algo_f in algos_conf_fs:
            try:
                algo = self.load_yaml_file(algo_f)
            except Exception as e:
                warnings.warn(
                    f"Could not load YAML config {algo_f} due to {e}"
                )
                continue
            if allowed_algos and algo["name"] not in allowed_algos:
                continue
            groups = algo.get("groups", {})
            if allowed_groups:
                groups = {
                    k: v for k, v in groups.items() if k in allowed_groups
                }
            algos_conf[algo["name"]] = {
                "groups": groups,
                "constraints": algo.get("constraints", {}),
            }
            if allowed_algo_groups and algo["name"] in allowed_algo_groups[0]:
                algos_conf[algo["name"]]["groups"].update(
                    {
                        group: algo["groups"][group]
                        for group in allowed_algo_groups[1]
                        if group in algo["groups"]
                    }
                )
        return algos_conf

    def validate_algorithm(
        self, algos_conf: dict, algo: str, gpu_present: bool
    ) -> bool:
        """
        Validate the algorithm based on the available hardware (GPU presence).

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
            True if the algorithm is valid for the current hardware
            configuration, False otherwise.
        """
        algos_conf_keys = set(algos_conf.keys())
        if gpu_present:
            return algo in algos_conf_keys
        return (
            algo in algos_conf_keys
            and algos_conf[algo]["requires_gpu"] is False
        )

    def find_executable(
        self, algos_conf: dict, algo: str, group: str, k: int, batch_size: int
    ) -> Tuple[str, str, Tuple[str, str]]:
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
            A tuple containing the executable name, the path to the executable,
            and the file name.
        """
        executable = algos_conf[algo]["executable"]
        file_name = (f"{algo},{group}", f"{algo},{group},k{k},bs{batch_size}")
        build_path = self.get_build_path(executable)
        if build_path:
            return executable, build_path, file_name
        raise FileNotFoundError(executable)

    def get_build_path(self, executable: str) -> Optional[str]:
        """
        Get the build path for the given executable.

        Parameters
        ----------
        executable : str
            The name of the executable.

        Returns
        -------
        Optional[str]
            The build path for the executable, if found.
        """

        devcontainer_path = "/home/coder/cuvs/cpp/build/latest/bench/ann"
        if os.path.exists(devcontainer_path):
            devc_executable = os.path.join(devcontainer_path, executable)
            print(f"-- Detected devcontainer artifact {devc_executable}.")
            return devc_executable

        build_path = os.getenv("CUVS_HOME")
        if build_path:
            build_path = os.path.join(
                build_path, "cpp", "build", "release", executable
            )
            if os.path.exists(build_path):
                print(f"-- Using cuVS bench from repository in {build_path}.")
                return build_path

        conda_path = os.getenv("CONDA_PREFIX")
        if conda_path:
            conda_executable = os.path.join(
                conda_path, "bin", "ann", executable
            )
            if os.path.exists(conda_executable):
                print("-- Using cuVS bench found in conda environment.")
                return conda_executable

        return None

    def prepare_indexes(
        self,
        group_conf: dict,
        algo: str,
        group: str,
        conf_file: dict,
        algos_conf: dict,
        dataset_path: str,
        dataset: str,
        count: int,
        batch_size: int,
        tune_mode: bool = False,
        tune_build_params: dict = None,
        tune_search_params: dict = None,
    ) -> list:
        """
        Prepare the index configurations for the given algorithm and group.

        Parameters
        ----------
        group_conf : dict
            The configuration for the algorithm group.
        algo : str
            The name of the algorithm.
        group : str
            The name of the group.
        conf_file : dict
            The main configuration file.
        dataset_path : str
            The path to the dataset directory.
        dataset : str
            The name of the dataset.
        count : int
            The number of nearest neighbors to search for.
        batch_size : int
            The size of each batch for processing.
        tune_mode : bool
            If True, use exact params from Optuna instead of Cartesian product.
        tune_build_params : dict
            Exact build parameters from Optuna (e.g., {"nlist": 5347}).
        tune_search_params : dict
            Exact search parameters from Optuna (e.g., {"nprobe": 73}).

        Returns
        -------
        list
            A list of index configurations.
            - Sweep mode: Multiple indexes (Cartesian product)
            - Tune mode: Single index (Optuna params)
        """
        indexes = []
        build_params = group_conf.get("build", {})
        search_params = group_conf.get("search", {})

        if tune_mode and tune_build_params is not None:
            # TUNE MODE: Use exact params from Optuna (single config)
            param_names = list(tune_build_params.keys())
            all_build_params = [tuple(tune_build_params.values())]
            # For search params, use tune_search_params keys/values
            if tune_search_params:
                search_param_names = tuple(tune_search_params.keys())
                search_param_lists = [[v] for v in tune_search_params.values()]
            else:
                search_param_names, search_param_lists = [], []
        else:
            # SWEEP MODE: Cartesian product of all YAML params
            all_build_params = itertools.product(*build_params.values())
            search_param_names, search_param_lists = (
                zip(*search_params.items()) if search_params else ([], [])
            )
            param_names = list(build_params.keys())
        for params in all_build_params:
            # Build the build_param dict using appropriate keys
            if tune_mode and tune_build_params is not None:
                # Tune mode: use exact params from Optuna
                index = {
                    "algo": algo,
                    "build_param": tune_build_params.copy(),
                }
            else:
                # Sweep mode: use YAML param names
                index = {
                    "algo": algo,
                    "build_param": dict(zip(param_names, params)),
                }

            # Build index name from params
            index_name = f"{algo}_{group}" if group != "base" else f"{algo}"
            for name, val in zip(param_names, params):
                index_name += f".{name}{val}"

            # Skip constraint validation in tune mode (Optuna handles bounds)
            if not tune_mode:
                if not self.validate_constraints(
                    algos_conf,
                    algo,
                    "build",
                    index["build_param"],
                    None,
                    conf_file["dataset"].get("dims"),
                    count,
                    batch_size,
                ):
                    continue

            index_filename = (
                index_name if len(index_name) < 128 else str(hash(index_name))
            )
            index["name"] = index_name
            index["file"] = os.path.join(
                dataset_path, dataset, "index", index_filename
            )

            # Handle search params
            if tune_mode and tune_search_params is not None:
                # Tune mode: use exact search params from Optuna (as single-item list)
                index["search_params"] = [tune_search_params.copy()]
            else:
                # Sweep mode: validate and expand search params
                index["search_params"] = self.validate_search_params(
                    itertools.product(*search_param_lists),
                    search_param_names,
                    index["build_param"],
                    algo,
                    group_conf,
                    algos_conf,
                    conf_file,
                    count,
                    batch_size,
                )

            if index["search_params"]:
                indexes.append(index)
        return indexes

    def validate_search_params(
        self,
        all_search_params,
        search_param_names,
        build_params,
        algo,
        group_conf,
        algos_conf,
        conf_file,
        count,
        batch_size,
    ) -> list:
        """
        Validate and prepare the search parameters for the given algorithm
        and group.

        Parameters
        ----------
        all_search_params : itertools.product
            The Cartesian product of search parameter values.
        search_param_names : list
            The names of the search parameters.
        algo : str
            The name of the algorithm.
        group_conf : dict
            The configuration for the algorithm group.
        conf_file : dict
            The main configuration file.
        count : int
            The number of nearest neighbors to search for.
        batch_size : int
            The size of each batch for processing.

        Returns
        -------
        list
            A list of validated search parameters.
        """
        search_params_list = []
        for search_params in all_search_params:
            search_dict = dict(zip(search_param_names, search_params))
            if self.validate_constraints(
                algos_conf,
                algo,
                "search",
                search_dict,
                build_params,
                conf_file["dataset"].get("dims"),
                count,
                batch_size,
            ):
                search_params_list.append(search_dict)
        return search_params_list

    def validate_constraints(
        self,
        algos_conf: Dict[str, Any],
        algo: str,
        constraint_type: str,
        param: Dict[str, Any],
        build_param: dict,
        dims: Any,
        k: Optional[int],
        batch_size: Optional[int],
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
        k : Optional[int]
            The number of nearest neighbors to search for.
        batch_size : Optional[int]
            The size of each batch for processing.

        Returns
        -------
        bool
            True if the constraints are valid, False otherwise.

        Raises
        ------
        ValueError
            If `dims` are needed for build constraints but not specified in the
            dataset configuration.
        """
        if constraint_type in algos_conf[algo]["constraints"]:
            importable = algos_conf[algo]["constraints"][constraint_type]
            module, func = (
                ".".join(importable.split(".")[:-1]),
                importable.split(".")[-1],
            )
            validator = import_module(module)
            constraints_func = getattr(validator, func)
            if constraint_type == "build":
                return constraints_func(param, dims)
            else:
                return constraints_func(param, build_param, k, batch_size)
        return True
