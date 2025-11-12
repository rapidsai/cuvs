#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Base classes and data structures for the cuvs-bench plugin system.

This module defines the abstract interface that all benchmark backends must implement,
along with standardized data structures for datasets and results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np


@dataclass
class Dataset:
    """
    Dataset representation for benchmarking.
    
    Attributes
    ----------
    name : str
        Dataset name (e.g., "glove-100-inner")
    base_vectors : np.ndarray
        Base vectors for index building, shape (n_vectors, dims)
    query_vectors : np.ndarray
        Query vectors for search, shape (n_queries, dims)
    groundtruth_neighbors : Optional[np.ndarray]
        Ground truth neighbor IDs, shape (n_queries, k_gt)
    groundtruth_distances : Optional[np.ndarray]
        Ground truth distances, shape (n_queries, k_gt)
    distance_metric : str
        Distance metric ("euclidean", "inner_product", "cosine")
    metadata : Dict[str, Any]
        Additional dataset metadata
    """
    name: str
    base_vectors: np.ndarray
    query_vectors: np.ndarray
    groundtruth_neighbors: Optional[np.ndarray] = None
    groundtruth_distances: Optional[np.ndarray] = None
    distance_metric: str = "euclidean"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def dims(self) -> int:
        """Vector dimensionality."""
        return self.base_vectors.shape[1]
    
    @property
    def n_base(self) -> int:
        """Number of base vectors."""
        return self.base_vectors.shape[0]
    
    @property
    def n_queries(self) -> int:
        """Number of query vectors."""
        return self.query_vectors.shape[0]


@dataclass
class BuildResult:
    """
    Results from index building phase.
    
    Attributes
    ----------
    index_path : str
        Path to the built index
    build_time_seconds : float
        Time taken to build the index (seconds)
    index_size_bytes : int
        Size of the built index (bytes)
    algorithm : str
        Algorithm name (e.g., "cuvs_ivf_flat")
    build_params : Dict[str, Any]
        Build parameters used (e.g., {"nlist": 1024, "niter": 20})
    metadata : Dict[str, Any]
        Additional metrics (e.g., GPU time, CPU time, memory usage)
    success : bool
        Whether the build succeeded
    error_message : Optional[str]
        Error message if build failed
    """
    index_path: str
    build_time_seconds: float
    index_size_bytes: int
    algorithm: str
    build_params: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable format (Google Benchmark compatible).
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with benchmark results
        """
        return {
            "name": f"{self.algorithm}/build",
            "real_time": self.build_time_seconds,
            "time_unit": "s",
            "index_size": self.index_size_bytes,
            "success": self.success,
            **self.build_params,
            **self.metadata
        }


@dataclass
class SearchResult:
    """
    Results from search phase.
    
    Attributes
    ----------
    neighbors : np.ndarray
        Neighbor IDs returned by search, shape (n_queries, k)
    distances : np.ndarray
        Distances to neighbors, shape (n_queries, k)
    search_time_seconds : float
        Total search time (seconds)
    queries_per_second : float
        Query throughput (QPS)
    recall : float
        Recall@k metric (0.0 to 1.0)
    algorithm : str
        Algorithm name
    search_params : Dict[str, Any]
        Search parameters used (e.g., {"nprobe": 10})
    latency_percentiles : Optional[Dict[str, float]]
        Latency percentiles in milliseconds (p50, p95, p99)
    gpu_time_seconds : Optional[float]
        GPU execution time (if applicable)
    cpu_time_seconds : Optional[float]
        CPU execution time
    metadata : Dict[str, Any]
        Additional metrics
    success : bool
        Whether the search succeeded
    error_message : Optional[str]
        Error message if search failed
    """
    neighbors: np.ndarray
    distances: np.ndarray
    search_time_seconds: float
    queries_per_second: float
    recall: float
    algorithm: str
    search_params: Dict[str, Any]
    latency_percentiles: Optional[Dict[str, float]] = None
    gpu_time_seconds: Optional[float] = None
    cpu_time_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable format (Google Benchmark compatible).
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with benchmark results
        """
        result = {
            "name": f"{self.algorithm}/search",
            "real_time": self.search_time_seconds,
            "time_unit": "s",
            "items_per_second": self.queries_per_second,
            "Recall": self.recall,
            "success": self.success,
            **self.search_params,
            **self.metadata
        }
        
        if self.latency_percentiles:
            result.update(self.latency_percentiles)
        
        if self.gpu_time_seconds is not None:
            result["GPU"] = self.gpu_time_seconds
        
        if self.cpu_time_seconds is not None:
            result["cpu_time"] = self.cpu_time_seconds
        
        return result


class BenchmarkBackend(ABC):
    """
    Abstract base class for all benchmark backends.
    
    All backends must implement this interface to be compatible with
    the cuvs-bench orchestration system.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Backend-specific configuration (e.g., executable path for C++ backends,
        host/port for network backends)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize backend with configuration."""
        self.config = config
    
    @abstractmethod
    def build(
        self,
        dataset: Dataset,
        build_params: Dict[str, Any],
        index_path: Path,
        force: bool = False
    ) -> BuildResult:
        """
        Build an index from the given dataset.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset with base vectors and metadata
        build_params : Dict[str, Any]
            Algorithm-specific build parameters (e.g., {"nlist": 1024, "niter": 20})
        index_path : Path
            Path where the built index should be saved
        force : bool, optional
            If True, rebuild even if index exists; if False, skip if exists
            
        Returns
        -------
        BuildResult
            Build timing, index size, and metadata
            
        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses
        """
        pass
    
    @abstractmethod
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
        Search for nearest neighbors using the built index.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset with query vectors and ground truth
        search_params : Dict[str, Any]
            Algorithm-specific search parameters (e.g., {"nprobe": 10})
        index_path : Path
            Path to the built index
        k : int
            Number of neighbors to return per query
        batch_size : int, optional
            Number of queries to process at once (default: 10000)
        mode : str, optional
            "latency" (measure individual query latency with percentiles) or
            "throughput" (measure overall QPS) (default: "throughput")
            
        Returns
        -------
        SearchResult
            Search timing, results, and recall metrics
            
        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by subclasses
        """
        pass
    
    def initialize(self) -> None:
        """
        Initialize backend (called once before any build/search operations).
        
        Use this for:
        - Establishing persistent connections (for database backends)
        - Initializing CUDA contexts
        - Loading shared resources
        - Pre-allocating memory
        
        Default implementation does nothing. Override if needed.
        """
        pass
    
    def cleanup(self) -> None:
        """
        Clean up backend resources (called once after all benchmarks complete).
        
        Use this for:
        - Closing database connections
        - Releasing GPU memory
        - Deleting temporary files
        - Freeing resources
        
        Default implementation does nothing. Override if needed.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique backend name (e.g., 'cpp_gbench', 'milvus', 'qdrant').
        
        Used for logging and result organization.
        
        Returns
        -------
        str
            Backend name
            
        Raises
        ------
        NotImplementedError
            This is an abstract property that must be implemented by subclasses
        """
        pass
    
    @property
    def supports_gpu(self) -> bool:
        """
        Whether this backend uses GPU acceleration.
        
        Returns
        -------
        bool
            True if backend uses GPU, False otherwise (default)
        """
        return False
    
    @property
    def requires_network(self) -> bool:
        """
        Whether this backend requires network connectivity.
        
        Returns
        -------
        bool
            True if backend needs network (e.g., remote VDB), False otherwise (default)
        """
        return False

