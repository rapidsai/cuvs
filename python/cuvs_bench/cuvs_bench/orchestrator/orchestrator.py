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
from .config_loaders import ConfigLoader, BenchmarkConfig, DatasetConfig, IndexConfig
from .registry import get_backend_class, get_config_loader, list_backends


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
        loader_class = get_config_loader(backend_type)
        
        # Instantiate config loader
        self.config_loader = loader_class()
    
    def run_benchmark(
        self,
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
            Additional arguments passed to ConfigLoader.load()
            (e.g., dataset, dataset_path, algorithms for cpp_gbench)
        
        Returns
        -------
        List[Union[BuildResult, SearchResult]]
            List of all build and search results
        """
        if not build and not search:
            build, search = True, True
        
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
            # Note: C++ backend loads vectors from files, so we pass empty arrays
            base_vectors=np.array([]),
            query_vectors=np.array([]),
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

