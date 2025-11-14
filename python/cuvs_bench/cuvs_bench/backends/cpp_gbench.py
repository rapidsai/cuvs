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
from typing import Dict, Any
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
        - executable_path: str - Path to C++ benchmark executable
        - data_prefix: str - Prefix for dataset paths (default: "")
        - index_prefix: str - Prefix for index paths (default: "")
        - warmup_time: float - Warmup time in seconds (default: 1.0)
    
    Examples
    --------
    >>> config = {
    ...     "executable_path": "/path/to/CUVS_IVF_FLAT_ANN_BENCH",
    ...     "data_prefix": "data/",
    ...     "index_prefix": "index/"
    ... }
    >>> backend = CppGoogleBenchmarkBackend(config)
    >>> result = backend.build(dataset, {"nlist": 1024}, Path("index/test"))
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize C++ benchmark backend."""
        super().__init__(config)
        
        self.executable_path = Path(config["executable_path"])
        self.data_prefix = config.get("data_prefix", "")
        self.index_prefix = config.get("index_prefix", "")
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
        force: bool = False
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
            Whether to rebuild if index exists
            
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
                algorithm=self.name,
                build_params=build_params,
                metadata={"skipped": True, "reason": "index_exists"},
                success=True
            )
        
        # Create temporary JSON config (Google Benchmark format)
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.json', 
            delete=False,
            dir='.'
        ) as f:
            config = {
                "dataset": {
                    "name": dataset.name,
                    "base_file": f"{dataset.name}/base.fbin",
                    "distance": dataset.distance_metric
                },
                "index": [{
                    "name": self.name,
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
            f"--index_prefix={self.index_prefix}",
            "--benchmark_out_format=json",
            "--benchmark_counters_tabular=true",
            f"--benchmark_out={temp_output_json}",
        ]
        
        if force:
            cmd.append("--force")
        
        cmd.append(temp_config_path)
        
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
                algorithm=self.name,
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
                algorithm=self.name,
                build_params=build_params,
                success=False,
                error_message=f"Build failed: {e.stderr}"
            )
        
        except Exception as e:
            return BuildResult(
                index_path=str(index_path),
                build_time_seconds=time.perf_counter() - start_time,
                index_size_bytes=0,
                algorithm=self.name,
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
        mode: str = "throughput"
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
            Number of neighbors to return
        batch_size : int
            Number of queries per batch
        mode : str
            "latency" or "throughput"
            
        Returns
        -------
        SearchResult
            Search timing, recall, and QPS
        """
        # Create temporary JSON config
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.json', 
            delete=False,
            dir='.'
        ) as f:
            config = {
                "dataset": {
                    "name": dataset.name,
                    "query_file": f"{dataset.name}/query.fbin",
                    "groundtruth_neighbors_file": f"{dataset.name}/groundtruth.neighbors.ibin",
                    "distance": dataset.distance_metric
                },
                "search_basic_param": {
                    "k": k,
                    "batch_size": batch_size
                },
                "index": [{
                    "name": self.name,
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
            f"--index_prefix={self.index_prefix}",
            f"--mode={mode}",
            f"--benchmark_min_warmup_time={self.warmup_time}",
            f"--override_kv=k:{k}",
            f"--override_kv=n_queries:{batch_size}",
            "--benchmark_out_format=json",
            "--benchmark_counters_tabular=true",
            f"--benchmark_out={output_json}",
            temp_config_path
        ]
        
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
                algorithm=self.name,
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
                algorithm=self.name,
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
                algorithm=self.name,
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
    def name(self) -> str:
        """
        Extract algorithm name from executable path.
        
        E.g., "CUVS_IVF_FLAT_ANN_BENCH" → "cuvs_ivf_flat"
        """
        exec_name = self.executable_path.stem
        # Remove _ANN_BENCH suffix and convert to lowercase
        name = exec_name.replace("_ANN_BENCH", "").lower()
        return name
    
    @property
    def supports_gpu(self) -> bool:
        """Check if this backend uses GPU."""
        name_lower = self.name.lower()
        return "cuvs" in name_lower or "faiss_gpu" in name_lower

