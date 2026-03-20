#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""
Elasticsearch GPU HTTP backend for cuvs-bench.

Install with: pip install cuvs-bench[elastic]

Build params (index_options): type, m, ef_construction.
  type: hnsw, int8_hnsw, int4_hnsw, bbq_hnsw (per ES-GPU-API-REFERENCE.md)
  similarity: l2_norm, cosine, max_inner_product (overrides dataset distance)
Index settings: number_of_shards, number_of_replicas, use_gpu, vector_field.
Search params (knn): num_candidates, vector_field.
"""

import itertools
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from cuvs_bench.backends.base import BenchmarkBackend, BuildResult, Dataset, SearchResult
from cuvs_bench.backends.registry import register_backend, register_config_loader
from cuvs_bench.orchestrator.config_loaders import (
    BenchmarkConfig,
    ConfigLoader,
    DatasetConfig,
    IndexConfig,
)


def _load_fbin(path: Path) -> np.ndarray:
    """Load big-ann-bench fbin format (header: n_rows, n_cols as uint32, then float32)."""
    with open(path, "rb") as f:
        n_rows, n_cols = np.fromfile(f, dtype=np.uint32, count=2)
        data = np.fromfile(f, dtype=np.float32).reshape(n_rows, n_cols)
    return data


def _load_ibin(path: Path) -> np.ndarray:
    """Load big-ann-bench ibin format (header: shape as uint32, then int32)."""
    with open(path, "rb") as f:
        shape = np.fromfile(f, dtype=np.uint32, count=2)
        data = np.fromfile(f, dtype=np.int32).reshape(shape[0], shape[1])
    return data


def _distance_to_similarity(distance: str) -> str:
    """Map cuvs-bench distance metric to ES dense_vector similarity."""
    m = {"euclidean": "l2_norm", "inner_product": "max_inner_product", "cosine": "cosine"}
    return m.get(distance, "l2_norm")


# Defaults for index creation when not specified in config
_DEFAULT_INDEX_TYPE = "hnsw"
_DEFAULT_M = 16
_DEFAULT_EF_CONSTRUCTION = 100
_DEFAULT_NUM_SHARDS = 1
_DEFAULT_NUM_REPLICAS = 0
_DEFAULT_USE_GPU = True
_DEFAULT_VECTOR_FIELD = "embedding"
_DEFAULT_NUM_CANDIDATES = 100

# Build params passed through from config (index_options + index settings + similarity)
_BUILD_PARAM_KEYS = (
    "type",
    "m",
    "ef_construction",
    "similarity",
    "number_of_shards",
    "number_of_replicas",
    "use_gpu",
    "vector_field",
)
# Search params passed through from config (knn)
_SEARCH_PARAM_KEYS = ("num_candidates", "vector_field")


class ElasticBackend(BenchmarkBackend):
    """Elasticsearch GPU backend for vector benchmarking."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client: Optional[Elasticsearch] = None

    @property
    def algo(self) -> str:
        """Algorithm name from config (e.g. elastic_hnsw, elastic_int8_hnsw)."""
        index_type = self.config.get("type", _DEFAULT_INDEX_TYPE)
        return f"elastic_{index_type}"

    def _get_client(self) -> Elasticsearch:
        if self._client is None:
            host = self.config.get("host", "localhost")
            port = self.config.get("port", 9200)
            kwargs: Dict[str, Any] = {"hosts": [f"http://{host}:{port}"]}
            basic_auth = self.config.get("basic_auth")
            if basic_auth is not None:
                if isinstance(basic_auth, (list, tuple)) and len(basic_auth) >= 2:
                    kwargs["basic_auth"] = (str(basic_auth[0]), str(basic_auth[1]))
                elif isinstance(basic_auth, str) and ":" in basic_auth:
                    user, _, passwd = basic_auth.partition(":")
                    kwargs["basic_auth"] = (user, passwd)
            self._client = Elasticsearch(**kwargs)
        return self._client

    def cleanup(self) -> None:
        """Close Elasticsearch client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def requires_network(self) -> bool:
        """Elasticsearch backend requires network connectivity."""
        return True

    def _check_network_available(self) -> bool:
        """Verify Elasticsearch is reachable."""
        try:
            return self._get_client().ping()
        except Exception:
            return False

    def build(
        self,
        dataset: Dataset,
        indexes: List[IndexConfig],
        force: bool = False,
        dry_run: bool = False,
    ) -> BuildResult:
        """Build ES index from dataset vectors (fbin)."""
        if dry_run:
            return BuildResult(
                index_path="",
                build_time_seconds=0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params={},
                success=True,
            )

        skip_reason = self._pre_flight_check()
        if skip_reason:
            idx = indexes[0] if indexes else None
            return BuildResult(
                index_path=idx.file if idx else "",
                build_time_seconds=0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params=dict(idx.build_param or {}) if idx else {},
                success=True,
                metadata={"skipped": True, "reason": skip_reason},
            )

        base_file = dataset.base_file
        if not base_file:
            return BuildResult(
                index_path="",
                build_time_seconds=0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params={},
                success=False,
                error_message="base_file is required for Elasticsearch backend",
            )

        data_prefix = self.config.get("data_prefix", "")
        base_path = Path(data_prefix) / base_file
        if not base_path.exists():
            return BuildResult(
                index_path="",
                build_time_seconds=0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params={},
                success=False,
                error_message=f"Base file not found: {base_path}",
            )

        index_name = self.config.get("index_name", "cuvs_bench_vectors")
        dims = dataset.dims or self.config.get("dims")
        if not dims:
            return BuildResult(
                index_path="",
                build_time_seconds=0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params={},
                success=False,
                error_message="dims is required",
            )

        idx = indexes[0] if indexes else None
        build_params = dict(idx.build_param or {}) if idx else {}
        for k, v in self.config.items():
            if k not in build_params and k in _BUILD_PARAM_KEYS:
                build_params[k] = v
        # similarity: from config, or derive from dataset distance
        similarity = build_params.get("similarity") or _distance_to_similarity(
            getattr(dataset, "distance_metric", None) or "euclidean"
        )

        try:
            vectors = _load_fbin(base_path)
            n_vectors = len(vectors)
            client = self._get_client()

            vector_field = build_params.get("vector_field", _DEFAULT_VECTOR_FIELD)
            index_type = build_params.get("type", _DEFAULT_INDEX_TYPE)
            m = build_params.get("m", _DEFAULT_M)
            ef_construction = build_params.get(
                "ef_construction", _DEFAULT_EF_CONSTRUCTION
            )
            num_shards = build_params.get("number_of_shards", _DEFAULT_NUM_SHARDS)
            num_replicas = build_params.get(
                "number_of_replicas", _DEFAULT_NUM_REPLICAS
            )
            use_gpu = build_params.get("use_gpu", _DEFAULT_USE_GPU)

            settings: Dict[str, Any] = {
                "number_of_shards": num_shards,
                "number_of_replicas": num_replicas,
            }
            if use_gpu:
                settings["index"] = {"vectors.indexing.use_gpu": True}

            index_options: Dict[str, Any] = {
                "type": index_type,
                "m": m,
                "ef_construction": ef_construction,
            }

            index_config: Dict[str, Any] = {
                "settings": settings,
                "mappings": {
                    "properties": {
                        vector_field: {
                            "type": "dense_vector",
                            "dims": dims,
                            "index": True,
                            "similarity": similarity,
                            "index_options": index_options,
                        },
                    },
                },
            }

            t0 = time.perf_counter()
            if client.indices.exists(index=index_name):
                if not force:
                    stats = client.indices.stats(index=index_name)
                    index_size = stats["_all"]["primaries"]["store"]["size_in_bytes"]
                    return BuildResult(
                        index_path=index_name,
                        build_time_seconds=0,
                        index_size_bytes=index_size,
                        algorithm=self.algo,
                        build_params=build_params,
                        success=True,
                    )
                client.indices.delete(index=index_name)
            client.indices.create(index=index_name, body=index_config)

            chunk_size = 1000
            for i in range(0, n_vectors, chunk_size):
                chunk = vectors[i : i + chunk_size]
                actions = [
                    {
                        "_index": index_name,
                        "_id": str(i + j),
                        vector_field: vec.tolist(),
                    }
                    for j, vec in enumerate(chunk)
                ]
                bulk(client, actions, raise_on_error=True)

            client.indices.refresh(index=index_name)
            build_time = time.perf_counter() - t0

            stats = client.indices.stats(index=index_name)
            index_size = stats["_all"]["primaries"]["store"]["size_in_bytes"]

            return BuildResult(
                index_path=index_name,
                build_time_seconds=build_time,
                index_size_bytes=index_size,
                algorithm=self.algo,
                build_params=build_params,
                success=True,
            )
        except Exception as e:
            return BuildResult(
                index_path="",
                build_time_seconds=0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params={},
                success=False,
                error_message=str(e),
            )

    def search(
        self,
        dataset: Dataset,
        indexes: List[IndexConfig],
        k: int = 10,
        batch_size: int = 10000,
        mode: str = "latency",
        force: bool = False,
        search_threads: Optional[int] = None,
        dry_run: bool = False,
    ) -> SearchResult:
        """Run kNN search and compute recall."""
        if dry_run:
            return SearchResult(
                neighbors=np.empty((0, k)),
                distances=np.empty((0, k)),
                search_time_ms=0,
                queries_per_second=0,
                recall=0,
                algorithm=self.algo,
                search_params=[],
                success=True,
            )

        skip_reason = self._pre_flight_check()
        if skip_reason:
            idx = indexes[0] if indexes else None
            return SearchResult(
                neighbors=np.empty((0, k)),
                distances=np.empty((0, k)),
                search_time_ms=0,
                queries_per_second=0,
                recall=0,
                algorithm=self.algo,
                search_params=idx.search_params if idx and idx.search_params else [],
                success=True,
                metadata={"skipped": True, "reason": skip_reason},
            )

        query_file = dataset.query_file
        gt_file = dataset.groundtruth_neighbors_file
        data_prefix = self.config.get("data_prefix", "")

        if not query_file:
            return SearchResult(
                neighbors=np.empty((0, k)),
                distances=np.empty((0, k)),
                search_time_ms=0,
                queries_per_second=0,
                recall=0,
                algorithm=self.algo,
                search_params=[],
                success=False,
                error_message="query_file is required",
            )

        query_path = Path(data_prefix) / query_file
        if not query_path.exists():
            return SearchResult(
                neighbors=np.empty((0, k)),
                distances=np.empty((0, k)),
                search_time_ms=0,
                queries_per_second=0,
                recall=0,
                algorithm=self.algo,
                search_params=[],
                success=False,
                error_message=f"Query file not found: {query_path}",
            )

        try:
            query_vectors = _load_fbin(query_path)
            n_queries = len(query_vectors)
            groundtruth = None
            if gt_file:
                gt_path = Path(data_prefix) / gt_file
                if gt_path.exists():
                    groundtruth = _load_ibin(gt_path)

            index_name = self.config.get("index_name", "cuvs_bench_vectors")
            search_params = {}
            if indexes and indexes[0] and indexes[0].search_params:
                search_params = dict(indexes[0].search_params[0] or {})
            for key, val in self.config.items():
                if key not in search_params and key in _SEARCH_PARAM_KEYS:
                    search_params[key] = val
            num_candidates = search_params.get(
                "num_candidates", _DEFAULT_NUM_CANDIDATES
            )
            vector_field = search_params.get(
                "vector_field", _DEFAULT_VECTOR_FIELD
            )

            neighbors_list: List[List[int]] = []
            latencies: List[float] = []

            for i, qv in enumerate(query_vectors):
                body = {
                    "knn": {
                        "field": vector_field,
                        "query_vector": qv.tolist(),
                        "k": k,
                        "num_candidates": num_candidates,
                    }
                }
                t0 = time.perf_counter()
                resp = self._get_client().search(
                    index=index_name, body=body, size=k
                )
                latencies.append((time.perf_counter() - t0) * 1000)
                hits = resp.get("hits", {}).get("hits", [])
                ids = [int(h["_id"]) for h in hits if "_id" in h]
                neighbors_list.append(ids[:k])

            search_time_ms = sum(latencies)
            qps = n_queries / (search_time_ms / 1000) if search_time_ms > 0 else 0

            recall = 0.0
            if groundtruth is not None and len(neighbors_list) <= len(groundtruth):
                correct = 0
                total = 0
                for q in range(len(neighbors_list)):
                    retrieved = set(neighbors_list[q])
                    gt_row = groundtruth[q, :k]
                    for gt_id in gt_row:
                        if int(gt_id) in retrieved:
                            correct += 1
                        total += 1
                recall = correct / total if total > 0 else 0

            max_len = max(len(r) for r in neighbors_list)
            neighbors_arr = np.array(
                [r + [-1] * (max_len - len(r)) for r in neighbors_list],
                dtype=np.int64,
            )
            distances_arr = np.zeros_like(neighbors_arr, dtype=np.float32)

            return SearchResult(
                neighbors=neighbors_arr,
                distances=distances_arr,
                search_time_ms=search_time_ms,
                queries_per_second=qps,
                recall=recall,
                algorithm=self.algo,
                search_params=[search_params],
                latency_percentiles={
                    "p50_ms": float(np.percentile(latencies, 50)),
                    "p95_ms": float(np.percentile(latencies, 95)),
                    "p99_ms": float(np.percentile(latencies, 99)),
                },
                success=True,
            )
        except Exception as e:
            return SearchResult(
                neighbors=np.empty((0, k)),
                distances=np.empty((0, k)),
                search_time_ms=0,
                queries_per_second=0,
                recall=0,
                algorithm=self.algo,
                search_params=[],
                success=False,
                error_message=str(e),
            )


def _get_cuvs_bench_config_path() -> str:
    """Get cuvs_bench config directory for shared datasets.yaml."""
    import cuvs_bench.orchestrator.config_loaders as _config_loaders

    mod_file = getattr(_config_loaders, "__file__", None)
    if mod_file:
        # config_loaders is at cuvs_bench/orchestrator/config_loaders.py
        # config is at cuvs_bench/config
        pkg_dir = Path(mod_file).resolve().parent.parent
        return str(pkg_dir / "config")
    import cuvs_bench

    if cuvs_bench.__file__:
        return os.path.join(os.path.dirname(cuvs_bench.__file__), "config")
    raise RuntimeError(
        "Cannot determine cuvs_bench config path. "
        "Ensure cuvs_bench is properly installed or on PYTHONPATH."
    )


def _get_elastic_config_path() -> str:
    """Get this package's config directory for elastic.yaml."""
    return os.path.join(os.path.dirname(__file__), "config")


class ElasticConfigLoader(ConfigLoader):
    """Config loader for Elasticsearch backend."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or _get_cuvs_bench_config_path()

    @property
    def backend_type(self) -> str:
        return "elastic"

    def load(
        self,
        dataset: str = "",
        dataset_path: str = "",
        count: int = 10,
        batch_size: int = 10000,
        host: str = "localhost",
        port: int = 9200,
        index_name: str = "cuvs_bench_vectors",
        basic_auth: Optional[Any] = None,
        algorithms: Optional[str] = None,
        subset_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[DatasetConfig, List[BenchmarkConfig]]:
        """Load Elasticsearch benchmark configuration."""
        tune_mode = kwargs.pop("_tune_mode", False)
        tune_build_params = kwargs.pop("_tune_build_params", None)
        tune_search_params = kwargs.pop("_tune_search_params", None)

        datasets_path = os.path.join(
            self.config_path, "datasets", "datasets.yaml"
        )
        try:
            with open(datasets_path) as f:
                datasets = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Datasets config not found: {datasets_path}"
            ) from None

        dataset_conf = next((d for d in datasets if d["name"] == dataset), None)
        if not dataset_conf:
            raise ValueError(f"Dataset '{dataset}' not found")

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

        if tune_mode and tune_build_params is not None and tune_search_params is not None:
            algo_name = algorithms or "elastic_hnsw"
            # Extract type from algo (elastic_hnsw -> hnsw, elastic_int8_hnsw -> int8_hnsw)
            build_param = dict(tune_build_params)
            if "type" not in build_param and algo_name.startswith("elastic_"):
                build_param["type"] = algo_name.replace("elastic_", "", 1)
            index_config = IndexConfig(
                name=f"{algo_name}_tune",
                algo=algo_name,
                build_param=build_param,
                search_params=[tune_search_params],
                file="",
            )
            config = BenchmarkConfig(
                indexes=[index_config],
                backend_config={
                    "name": index_config.name,
                    "host": host,
                    "port": port,
                    "index_name": index_name,
                    "basic_auth": basic_auth,
                    "data_prefix": dataset_path,
                    "dims": dataset_config.dims,
                    **build_param,
                    **tune_search_params,
                },
            )
            return dataset_config, [config]

        elastic_config_dir = _get_elastic_config_path()
        elastic_algo_path = os.path.join(elastic_config_dir, "algos", "elastic.yaml")
        default_grp = {
            "build": {"m": [16], "ef_construction": [100]},
            "search": {"num_candidates": [100]},
        }
        if os.path.exists(elastic_algo_path):
            with open(elastic_algo_path) as f:
                algo_conf = yaml.safe_load(f)
        else:
            algo_conf = {"groups": {"base": default_grp}}

        groups = algo_conf.get("groups", {"base": default_grp})
        group_name = algorithms or "base"
        if group_name not in groups:
            raise ValueError(
                f"Algorithm group '{group_name}' not found in elastic.yaml. "
                f"Available: {list(groups.keys())}"
            )
        group_conf = groups[group_name]

        build_params = group_conf.get(
            "build", {"m": [16], "ef_construction": [100]}
        )
        search_params = group_conf.get("search", {"num_candidates": [100]})

        # Ensure all param values are lists for itertools.product
        def _to_list_values(d: Dict[str, Any]) -> Dict[str, List[Any]]:
            return {k: v if isinstance(v, list) else [v] for k, v in d.items()}

        build_params = _to_list_values(build_params)
        search_params = _to_list_values(search_params)

        build_combos = list(itertools.product(*build_params.values()))
        search_combos = list(itertools.product(*search_params.values()))
        build_keys = list(build_params.keys())
        search_keys = list(search_params.keys())

        benchmark_configs = []
        for bvals in build_combos:
            bdict = dict(zip(build_keys, bvals))
            for svals in search_combos:
                sdict = dict(zip(search_keys, svals))
                algo_name = f"elastic_{bdict.get('type', 'hnsw')}"
                index_config = IndexConfig(
                    name=f"{algo_name}_{group_name}",
                    algo=algo_name,
                    build_param=bdict,
                    search_params=[sdict],
                    file="",
                )
                config = BenchmarkConfig(
                    indexes=[index_config],
                    backend_config={
                        "name": index_config.name,
                        "host": host,
                        "port": port,
                        "index_name": index_name,
                        "basic_auth": basic_auth,
                        "data_prefix": dataset_path,
                        "dims": dataset_config.dims,
                        **bdict,
                        **sdict,
                    },
                )
                benchmark_configs.append(config)

        return dataset_config, benchmark_configs


def register() -> None:
    """Register Elasticsearch backend and config loader (called from entry point)."""
    register_backend("elastic", ElasticBackend)
    register_config_loader("elastic", ElasticConfigLoader)
