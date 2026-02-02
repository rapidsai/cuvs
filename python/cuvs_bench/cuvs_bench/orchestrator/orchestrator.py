#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""
Benchmark Orchestrator for cuvs-bench.

This module provides the BenchmarkOrchestrator class that coordinates
benchmark runs across different backends using the registry pattern.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..backends.base import BenchmarkBackend, Dataset, BuildResult, SearchResult
from ..backends.registry import get_backend_class, get_config_loader, list_backends
from .config_loaders import ConfigLoader, BenchmarkConfig, DatasetConfig, IndexConfig


class BenchmarkOrchestrator:
    """
    Orchestrator for running benchmarks using the pluggable backend system.
    
    This class coordinates benchmark runs by:
    1. Using a ConfigLoader to load backend-specific configurations
    2. Creating Backend instances from the registry
    3. Running build/search operations
    4. Collecting and returning results
    
    Parameters
    ----------
    backend_type : str
        Type of backend to use (e.g., 'cpp_gbench', 'milvus')
    **loader_kwargs
        Additional arguments passed to the ConfigLoader constructor
    
    Examples
    --------
    >>> # For C++ benchmarks
    >>> from cuvs_bench.orchestrator import BenchmarkOrchestrator
    >>> orchestrator = BenchmarkOrchestrator(backend_type="cpp_gbench")
    >>> results = orchestrator.run_benchmark(
    ...     dataset="glove-100-inner",
    ...     dataset_path="/data/",
    ...     algorithms="cuvs_ivf_flat",
    ...     build=True,
    ...     search=True
    ... )
    
    >>> # For Milvus benchmarks (when implemented)
    >>> orchestrator = BenchmarkOrchestrator(backend_type="milvus")
    >>> results = orchestrator.run_benchmark(
    ...     host="localhost",
    ...     port=19530,
    ...     ...
    ... )
    """
    
    def __init__(self, backend_type: str = "cpp_gbench"):
        """
        Initialize the orchestrator.
        
        Parameters
        ----------
        backend_type : str
            Type of backend to use
        """
        self.backend_type = backend_type
        
        # Get classes from registry
        self.backend_class = get_backend_class(backend_type)
        # Get the config loader for this specific backend
        loader_class = get_config_loader(backend_type)
        
        # Instantiate config loader
        self.config_loader = loader_class()
    
    def run_benchmark(
        self,
        mode: str = "sweep",
        constraints: dict = None,
        n_trials: int = None,
        build: bool = True,
        search: bool = True,
        force: bool = False,
        dry_run: bool = False,
        count: int = 10,
        batch_size: int = 10000,
        search_mode: str = "latency",
        search_threads: Optional[int] = None,
        **loader_kwargs
    ) -> List[Union[BuildResult, SearchResult]]:
        """
        Run benchmarks using the configured backend.
        
        Parameters
        ----------
        mode : str
            Benchmark mode: "sweep" (default, exhaustive Cartesian product) or
            "tune" (intelligent search using Optuna)
        constraints : dict
            For tune mode: optimization constraints. Metrics with "min" bounds
            are maximized, metrics with "max" bounds are hard limits.
            Example: {"recall": {"min": 0.95}, "latency": {"max": 10}}
        n_trials : int
            For tune mode: maximum number of Optuna trials. Ignored in sweep mode.
        build : bool
            Whether to build indices
        search : bool
            Whether to perform searches
        force : bool
            Whether to force rebuild even if index exists
        dry_run : bool
            Whether to perform a dry run without actual execution
        count : int
            Number of neighbors (k) to search for
        batch_size : int
            Batch size for search
        search_mode : str
            Search mode ('throughput' or 'latency')
        search_threads : Optional[int]
            Number of threads for search
        **loader_kwargs
            Backend-specific arguments passed to ConfigLoader.load().
            
            These vary by backend type to keep the orchestrator backend-agnostic:
            - cpp_gbench: dataset, dataset_path, algorithms, groups, algo_groups,
              dataset_configuration, algorithm_configuration, subset_size
            - milvus (future): host, port, collection, api_key
            - qdrant (future): host, api_key, collection
        
        Returns
        -------
        List[Union[BuildResult, SearchResult]]
            List of all build and search results
        """
        # Validate mode parameter
        valid_modes = ("sweep", "tune")
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")
        
        # If build and search are not provided, set them to True by default
        if not build and not search:
            build, search = True, True
        
        # Branch based on mode
        if mode == "sweep":
            return self._run_sweep(
                build=build,
                search=search,
                force=force,
                dry_run=dry_run,
                count=count,
                batch_size=batch_size,
                search_mode=search_mode,
                search_threads=search_threads,
                **loader_kwargs
            )
        else:  # mode == "tune"
            return self._run_tune(
                constraints=constraints,
                n_trials=n_trials,
                build=build,
                search=search,
                force=force,
                dry_run=dry_run,
                count=count,
                batch_size=batch_size,
                search_mode=search_mode,
                search_threads=search_threads,
                **loader_kwargs
            )
    
    def _run_sweep(
        self,
        build: bool,
        search: bool,
        force: bool,
        dry_run: bool,
        count: int,
        batch_size: int,
        search_mode: str,
        search_threads: Optional[int],
        **loader_kwargs
    ) -> List[Union[BuildResult, SearchResult]]:
        """
        Run exhaustive sweep over all parameter combinations.
        
        This is the default mode that runs all configurations from the
        Cartesian product of build/search parameters defined in YAML.
        """
        # Load configurations using the backend-specific loader
        dataset_config, benchmark_configs = self.config_loader.load(
            count=count,
            batch_size=batch_size,
            **loader_kwargs
        )
        
        # Create Dataset object from config
        bench_dataset = self._create_dataset(dataset_config)
        
        # Collect results
        results: List[Union[BuildResult, SearchResult]] = []
        
        # Run benchmarks
        # Each config contains ALL indexes for one executable (matches runners.py)
        for config in benchmark_configs:
            # Create backend instance
            backend = self.backend_class(config.backend_config)
            
            if build:
                # Pass ALL indexes at once - ONE C++ command builds all
                build_result = backend.build(
                    dataset=bench_dataset,
                    indexes=config.indexes,
                    force=force,
                    dry_run=dry_run
                )
                results.append(build_result)
                
                if not build_result.success:
                    print(f"Build failed for {config.index_name}: {build_result.error_message}")
                    continue
            
            if search:
                # Pass ALL indexes at once - ONE C++ command searches all
                # Each index has its own search_params list
                # Total benchmarks = sum(len(idx.search_params) for idx in indexes)
                search_result = backend.search(
                    dataset=bench_dataset,
                    indexes=config.indexes,
                    k=count,
                    batch_size=batch_size,
                    mode=search_mode,
                    force=force,
                    search_threads=search_threads,
                    dry_run=dry_run
                )
                results.append(search_result)
                
                if not search_result.success:
                    print(f"Search failed for {config.index_name}: {search_result.error_message}")
        
        return results
    
    def _run_tune(
        self,
        constraints: dict,
        n_trials: int,
        build: bool,
        search: bool,
        force: bool,
        dry_run: bool,
        count: int,
        batch_size: int,
        search_mode: str,
        search_threads: Optional[int],
        **loader_kwargs
    ) -> List[Union[BuildResult, SearchResult]]:
        """
        Run intelligent parameter tuning using Optuna.
        
        Instead of exhaustive sweep, uses Optuna to intelligently explore
        the parameter space and find optimal configurations based on
        the provided constraints.
        
        Parameters
        ----------
        constraints : dict
            Optimization constraints. Metrics with "min" bounds are maximized,
            metrics with "max" bounds are hard limits.
        n_trials : int
            Maximum number of Optuna trials.
        """
        raise NotImplementedError("Tune mode not yet implemented")
    
    def _create_dataset(self, dataset_config: DatasetConfig) -> Dataset:
        """
        Create a Dataset object from DatasetConfig.
        
        Parameters
        ----------
        dataset_config : DatasetConfig
            Dataset configuration
            
        Returns
        -------
        Dataset
            Dataset object for use with backends
        """
        return Dataset(
            name=dataset_config.name,
            # C++ backend loads vectors from files; other backends may need actual arrays
            base_vectors=np.empty((0, 0)),
            query_vectors=np.empty((0, 0)),
            # Ground truth arrays - used by Python-native backends for recall calculation
            groundtruth_neighbors=None,
            groundtruth_distances=None,
            # File paths - used by C++ backend
            base_file=dataset_config.base_file,
            query_file=dataset_config.query_file,
            groundtruth_neighbors_file=dataset_config.groundtruth_neighbors_file,
            distance_metric=dataset_config.distance,
            metadata={"subset_size": dataset_config.subset_size}
        )
    
    @staticmethod
    def available_backends() -> List[str]:
        """
        List all available backend types.
        
        Returns
        -------
        List[str]
            List of registered backend names
        """
        return list(list_backends().keys())


# ============================================================================
# Standalone function for backward compatibility
# ============================================================================

def run_benchmark(
    subset_size: int = None,
    count: int = 10,
    batch_size: int = 10000,
    dataset_configuration: Optional[str] = None,
    configuration: Optional[str] = None,
    dataset: str = None,
    dataset_path: str = None,
    build: Optional[bool] = None,
    search: Optional[bool] = None,
    algorithms: Optional[str] = None,
    groups: Optional[str] = None,
    algo_groups: Optional[str] = None,
    force: bool = False,
    search_mode: str = "latency",
    search_threads: Optional[int] = None,
    dry_run: bool = False,
    data_export: bool = False,
    backend_type: str = "cpp_gbench",
) -> List[Union[BuildResult, SearchResult]]:
    """
    Standalone function for backward compatibility with run.py interface.
    
    Parameters
    ----------
    subset_size : int
        The subset size of the dataset.
    count : int
        The number of nearest neighbors to search for.
    batch_size : int
        The size of each batch for processing.
    dataset_configuration : Optional[str]
        Path to the dataset configuration file.
    configuration : Optional[str]
        Path to the algorithm configuration directory or file.
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
    data_export : bool
        Whether to export data (unused, kept for compatibility).
    backend_type : str
        Type of backend to use (default: 'cpp_gbench')
    
    Returns
    -------
    List[Union[BuildResult, SearchResult]]
        List of all build and search results.
    """
    orchestrator = BenchmarkOrchestrator(backend_type=backend_type)
    
    return orchestrator.run_benchmark(
        build=build if build is not None else True,
        search=search if search is not None else True,
        force=force,
        dry_run=dry_run,
        count=count,
        batch_size=batch_size,
        search_mode=search_mode,
        search_threads=search_threads,
        # ConfigLoader-specific kwargs (for cpp_gbench)
        dataset=dataset,
        dataset_path=dataset_path,
        dataset_configuration=dataset_configuration,
        configuration=configuration,
        algorithms=algorithms,
        groups=groups,
        algo_groups=algo_groups,
        subset_size=subset_size,
    )

