#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
C++ Google Benchmark backend for cuvs-bench.

This backend wraps existing C++ Google Benchmark executables to maintain
backward compatibility with the current cuvs-bench infrastructure.
"""

import subprocess
import json
import tempfile
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from .base import BenchmarkBackend, Dataset, BuildResult, SearchResult


class CppGoogleBenchmarkBackend(BenchmarkBackend):
    """
    Backend for existing C++ Google Benchmark executables.
    
    This wraps the current cuvs_bench_cpp() logic into the plugin interface,
    ensuring 100% backward compatibility with existing C++ benchmarks.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary with:
        - name: str - User-defined name for this index configuration (required)
        - executable_path: str - Path to C++ benchmark executable
        - data_prefix: str - Prefix for dataset paths (default: "")
        - warmup_time: float - Warmup time in seconds (default: 1.0)
    
    Examples
    --------
    >>> config = {
    ...     "name": "ivf_flat_experiment",
    ...     "executable_path": "/path/to/CUVS_IVF_FLAT_ANN_BENCH",
    ...     "data_prefix": "data/"
    ... }
    >>> backend = CppGoogleBenchmarkBackend(config)
    >>> print(backend.name)  # "ivf_flat_experiment" (user-defined)
    >>> print(backend.algo)  # "cuvs_ivf_flat" (from executable)
    >>> result = backend.build(dataset, {"nlist": 1024}, Path("index/test"))
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize C++ benchmark backend."""
        super().__init__(config)
        
        self.executable_path = Path(config["executable_path"])
        self.data_prefix = config.get("data_prefix", "")
        self.warmup_time = config.get("warmup_time", 1.0)
        
        if not self.executable_path.exists():
            raise FileNotFoundError(
                f"C++ benchmark executable not found: {self.executable_path}"
            )
    
    def build(
        self,
        dataset: Dataset,
        build_params: Dict[str, Any],
        index_path: Path,
        force: bool = False,
        dry_run: bool = False
    ) -> BuildResult:
        """
        Build index using C++ Google Benchmark executable.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset with base vectors
        build_params : Dict[str, Any]
            Build parameters (e.g., {"nlist": 1024, "niter": 20})
        index_path : Path
            Path to save the built index
        force : bool
            Whether to force the execution regardless of existing results.
        dry_run : bool
            Whether to perform a dry run without actual execution.
            
        Returns
        -------
        BuildResult
            Build timing and metadata
        """
        # Check if index exists and skip if not forcing
        if index_path.exists() and not force:
            return BuildResult(
                index_path=str(index_path),
                build_time_seconds=0.0,
                index_size_bytes=index_path.stat().st_size,
                algorithm=self.algo,
                build_params=build_params,
                metadata={"skipped": True, "reason": "index_exists"},
                success=True
            )

        # Note: runners.py doesn't validate and lets C++ fail. We validate here for
        # better Python-side error messages.
        # C++ requires: name, base_file, query_file, distance (see conf.hpp parse_dataset)
        # C++ optional: groundtruth_neighbors_file
        if not dataset.base_file:
            raise ValueError("dataset.base_file is required (C++ parser requires it)")
        if not dataset.query_file:
            raise ValueError("dataset.query_file is required (C++ parser requires it)")
        
        # Create temporary JSON config (Google Benchmark format)
        # Structure matches runners.py temp_conf exactly
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.json', 
            delete=False,
            dir='.'
        ) as f:
            dataset_config = {
                "name": dataset.name,
                "base_file": dataset.base_file,
                "query_file": dataset.query_file,
                "distance": dataset.distance_metric
            }
            # groundtruth_neighbors_file is optional in C++
            if dataset.groundtruth_neighbors_file:
                dataset_config["groundtruth_neighbors_file"] = dataset.groundtruth_neighbors_file
            
            config = {
                "dataset": dataset_config,
                "search_basic_param": {
                    "k": 10,
                    "batch_size": 10000
                },
                "index": [{
                    "name": self.name,
                    "algo": self.algo,
                    "build_param": build_params,
                    "file": str(index_path)
                }]
            }
            json.dump(config, f, indent=2)
            temp_config_path = f.name
        
        # Prepare output directory and file
        output_dir = index_path.parent / "result" / "build"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json = output_dir / f"{index_path.name}.build.json"
        temp_output_json = output_dir / f"{output_json.name}.lock"
        
        # Construct C++ command
        cmd = [
            str(self.executable_path),
            "--build",
            f"--data_prefix={self.data_prefix}",
            "--benchmark_out_format=json",
            "--benchmark_counters_tabular=true",
            f"--benchmark_out={temp_output_json}",
        ]
        
        if force:
            cmd.append("--force")
        
        cmd.append(temp_config_path)
        
        # Dry run: print command and return without executing
        if dry_run:
            print(f"Benchmark command for {index_path.name}:\n{' '.join(cmd)}\n")
            Path(temp_config_path).unlink(missing_ok=True)
            return BuildResult(
                index_path=str(index_path),
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params=build_params,
                metadata={"dry_run": True},
                success=True
            )
        
        # Execute subprocess
        start_time = time.perf_counter()
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            elapsed_time = time.perf_counter() - start_time
            
            # Parse Google Benchmark JSON output
            with open(temp_output_json) as f:
                gbench_results = json.load(f)
            
            # Merge with existing results (if any)
            self._merge_build_results(output_json, temp_output_json, gbench_results)
            
            # Extract build metrics
            benchmarks = gbench_results.get("benchmarks", [])
            if not benchmarks:
                raise ValueError("No benchmarks found in Google Benchmark output")
            
            benchmark = benchmarks[0]
            
            return BuildResult(
                index_path=str(index_path),
                build_time_seconds=benchmark.get("real_time", elapsed_time),
                index_size_bytes=index_path.stat().st_size if index_path.exists() else 0,
                algorithm=self.algo,
                build_params=build_params,
                metadata={
                    "cpu_time": benchmark.get("cpu_time"),
                    "gpu_time": benchmark.get("GPU"),
                    "context": gbench_results.get("context", {})
                },
                success=True
            )
        
        except subprocess.CalledProcessError as e:
            return BuildResult(
                index_path=str(index_path),
                build_time_seconds=time.perf_counter() - start_time,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params=build_params,
                success=False,
                error_message=f"Build failed: {e.stderr}"
            )
        
        except Exception as e:
            return BuildResult(
                index_path=str(index_path),
                build_time_seconds=time.perf_counter() - start_time,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params=build_params,
                success=False,
                error_message=f"Build error: {str(e)}"
            )
        
        finally:
            # Cleanup temporary files
            Path(temp_config_path).unlink(missing_ok=True)
            Path(temp_output_json).unlink(missing_ok=True)
    
    def search(
        self,
        dataset: Dataset,
        search_params: Dict[str, Any],
        index_path: Path,
        k: int,
        batch_size: int = 10000,
        mode: str = "throughput",
        search_threads: Optional[int] = None,
        dry_run: bool = False
    ) -> SearchResult:
        """
        Search using C++ Google Benchmark executable.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset with query vectors and ground truth
        search_params : Dict[str, Any]
            Search parameters (e.g., {"nprobe": 10})
        index_path : Path
            Path to the built index
        k : int
            The number of nearest neighbors to search for.
        batch_size : int
            The size of each batch for processing.
        mode : str
            The mode of search to perform ('latency' or 'throughput'),
            by default 'throughput'.
        search_threads : Optional[int]
            The number of threads to use for searching.
        dry_run : bool
            Whether to perform a dry run without actual execution.
            
        Returns
        -------
        SearchResult
            Search timing, recall, and QPS
        """
        # Note: runners.py doesn't validate and lets C++ fail. We validate here for
        # better Python-side error messages.
        # C++ requires: name, base_file, query_file, distance (see conf.hpp parse_dataset)
        # C++ optional: groundtruth_neighbors_file (but needed for recall calculation)
        if not dataset.base_file:
            raise ValueError("dataset.base_file is required (C++ parser requires it)")
        if not dataset.query_file:
            raise ValueError("dataset.query_file is required (C++ parser requires it)")
        
        # Create temporary JSON config
        # Structure matches runners.py temp_conf exactly
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.json', 
            delete=False,
            dir='.'
        ) as f:
            dataset_config = {
                "name": dataset.name,
                "base_file": dataset.base_file,
                "query_file": dataset.query_file,
                "distance": dataset.distance_metric
            }
            # groundtruth_neighbors_file is optional in C++, but needed for recall calculation
            if dataset.groundtruth_neighbors_file:
                dataset_config["groundtruth_neighbors_file"] = dataset.groundtruth_neighbors_file
            
            config = {
                "dataset": dataset_config,
                "search_basic_param": {
                    "k": k,
                    "batch_size": batch_size
                },
                "index": [{
                    "name": self.name,
                    "algo": self.algo,
                    "file": str(index_path),
                    "search_params": [search_params]
                }]
            }
            json.dump(config, f, indent=2)
            temp_config_path = f.name
        
        # Prepare output file
        output_dir = index_path.parent / "result" / "search"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json = output_dir / f"{index_path.name}.search.json"
        
        # Construct C++ command
        cmd = [
            str(self.executable_path),
            "--search",
            f"--data_prefix={self.data_prefix}",
            f"--mode={mode}",
            f"--benchmark_min_warmup_time={self.warmup_time}",
            f"--override_kv=k:{k}",
            f"--override_kv=n_queries:{batch_size}",
            "--benchmark_out_format=json",
            "--benchmark_counters_tabular=true",
            f"--benchmark_out={output_json}",
        ]
        
        if search_threads:
            cmd.append(f"--threads={search_threads}")
        
        cmd.append(temp_config_path)
        
        # Dry run: print command and return without executing
        if dry_run:
            print(f"Benchmark command for {index_path.name}:\n{' '.join(cmd)}\n")
            Path(temp_config_path).unlink(missing_ok=True)
            return SearchResult(
                neighbors=np.array([]),
                distances=np.array([]),
                search_time_seconds=0.0,
                queries_per_second=0.0,
                recall=0.0,
                algorithm=self.algo,
                search_params=search_params,
                metadata={"dry_run": True},
                success=True
            )
        
        # Execute subprocess
        start_time = time.perf_counter()
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600
            )
            elapsed_time = time.perf_counter() - start_time
            
            # Parse results
            with open(output_json) as f:
                gbench_results = json.load(f)
            
            benchmarks = gbench_results.get("benchmarks", [])
            if not benchmarks:
                raise ValueError("No benchmarks found in Google Benchmark output")
            
            benchmark = benchmarks[0]
            
            # Extract metrics
            recall = benchmark.get("Recall", 0.0)
            qps = benchmark.get("items_per_second", 0.0)
            search_time = benchmark.get("real_time", elapsed_time)
            
            # Note: C++ Google Benchmark doesn't return actual neighbors/distances
            # This is a limitation of the current system
            return SearchResult(
                neighbors=np.array([]),  # Not available from C++ benchmark
                distances=np.array([]),  # Not available from C++ benchmark
                search_time_seconds=search_time,
                queries_per_second=qps,
                recall=recall,
                algorithm=self.algo,
                search_params=search_params,
                gpu_time_seconds=benchmark.get("GPU"),
                cpu_time_seconds=benchmark.get("cpu_time"),
                metadata={
                    "latency_us": benchmark.get("Latency"),
                    "end_to_end": benchmark.get("end_to_end"),
                    "context": gbench_results.get("context", {})
                },
                success=True
            )
        
        except subprocess.CalledProcessError as e:
            return SearchResult(
                neighbors=np.array([]),
                distances=np.array([]),
                search_time_seconds=time.perf_counter() - start_time,
                queries_per_second=0.0,
                recall=0.0,
                algorithm=self.algo,
                search_params=search_params,
                success=False,
                error_message=f"Search failed: {e.stderr}"
            )
        
        except Exception as e:
            return SearchResult(
                neighbors=np.array([]),
                distances=np.array([]),
                search_time_seconds=time.perf_counter() - start_time,
                queries_per_second=0.0,
                recall=0.0,
                algorithm=self.algo,
                search_params=search_params,
                success=False,
                error_message=f"Search error: {str(e)}"
            )
        
        finally:
            # Cleanup temporary config
            Path(temp_config_path).unlink(missing_ok=True)
    
    def _merge_build_results(
        self, 
        output_json: Path, 
        temp_output_json: Path,
        temp_results: Dict
    ) -> None:
        """
        Merge temporary build results with existing results.
        
        This replicates the merge_build_files() logic from runners.py.
        """
        existing_results = {}
        
        # Load existing results if file exists
        if output_json.exists():
            try:
                with open(output_json) as f:
                    existing_results = json.load(f)
            except Exception as e:
                print(f"Warning: Error loading existing build file: {e}")
        
        # Get benchmarks from both files
        existing_benchmarks = existing_results.get("benchmarks", [])
        temp_benchmarks = temp_results.get("benchmarks", [])
        
        # Filter out failed builds (real_time == 0)
        valid_benchmarks = {
            b["name"]: b 
            for b in existing_benchmarks 
            if b.get("real_time", 0) > 0
        }
        
        # Add/update with temp benchmarks
        for bench in temp_benchmarks:
            if bench.get("real_time", 0) > 0:
                valid_benchmarks[bench["name"]] = bench
        
        # Write merged results
        temp_results["benchmarks"] = list(valid_benchmarks.values())
        with open(output_json, 'w') as f:
            json.dump(temp_results, f, indent=2)
    
    @property
    def algo(self) -> str:
        """
        Extract algorithm name from executable path.
        
        E.g., "CUVS_IVF_FLAT_ANN_BENCH" → "cuvs_ivf_flat"
        """
        exec_name = self.executable_path.stem
        # Remove _ANN_BENCH suffix and convert to lowercase
        algo = exec_name.replace("_ANN_BENCH", "").lower()
        return algo
    
    @property
    def supports_gpu(self) -> bool:
        """Check if this backend uses GPU."""
        algo_lower = self.algo.lower()
        return "cuvs" in algo_lower or "faiss_gpu" in algo_lower

