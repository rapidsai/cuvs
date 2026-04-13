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
Index settings: number_of_shards, number_of_replicas, vector_field.
  GPU indexing is configured at the node level (vectors.indexing.use_gpu in elasticsearch.yml).
Search params (knn): num_candidates, vector_field.
"""

import itertools
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


from .base import BenchmarkBackend, BuildResult, Dataset, SearchResult
from .registry import register_backend, register_config_loader
from ..orchestrator.config_loaders import (
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

_DEFAULT_VECTOR_FIELD = "embedding"
_DEFAULT_NUM_CANDIDATES = 100

_BUILD_PARAM_KEYS = (
    "type",
    "m",
    "ef_construction",
    "similarity",
    "number_of_shards",
    "number_of_replicas",
    "vector_field",
)
_SEARCH_PARAM_KEYS = ("num_candidates", "vector_field")


class ElasticBackend(BenchmarkBackend):
    """Elasticsearch GPU backend for vector benchmarking."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client: Optional["Elasticsearch"] = None

    @property
    def algo(self) -> str:
        """Algorithm name from config (e.g. elastic_hnsw, elastic_int8_hnsw)."""
        index_type = self.config.get("type", _DEFAULT_INDEX_TYPE)
        return f"elastic_{index_type}"

    def _get_client(self) -> "Elasticsearch":
        if self._client is None:
            try:
                from elasticsearch import Elasticsearch
            except ImportError as e:
                raise ImportError(
                    "`elasticsearch` is required for the Elasticsearch backend. "
                    "Install with: pip install cuvs-bench[elastic]"
                ) from e
            host = self.config.get("host", "localhost")
            port = self.config.get("port", 9200)
            scheme = self.config.get("scheme", "http")
            kwargs: Dict[str, Any] = {
                "hosts": [f"{scheme}://{host}:{port}"],
                "request_timeout": 300,  # 5 min for slow bulk ops / remote ES
            }
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
            return BuildResult(
                index_path="",
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params={},
                success=False,
                error_message=f"pre-flight check failed: {skip_reason}",
            )

        index_name = self.config.get("index_name", "cuvs_bench_vectors")
        idx = indexes[0] if indexes else None
        build_params = dict(idx.build_param or {}) if idx else {}
        for k, v in self.config.items():
            if k not in build_params and k in _BUILD_PARAM_KEYS:
                build_params[k] = v

        try:
            client = self._get_client()
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

        if not dataset.base_file:
            return BuildResult(
                index_path="",
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params={},
                success=False,
                error_message="base_file is required for Elasticsearch backend",
            )

        base_path = Path(dataset.base_file)
        if not base_path.exists():
            return BuildResult(
                index_path="",
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params={},
                success=False,
                error_message=f"Base file not found: {base_path}",
            )

        # similarity: from config, or derive from dataset distance
        similarity = build_params.get("similarity") or _distance_to_similarity(
            getattr(dataset, "distance_metric", None) or "euclidean"
        )

        try:
            vectors = _load_fbin(base_path)
            n_vectors = len(vectors)
            dims = vectors.shape[1]
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
            settings: Dict[str, Any] = {
                "number_of_shards": num_shards,
                "number_of_replicas": num_replicas,
            }
            # Note: GPU indexing is controlled at the node level via
            # vectors.indexing.use_gpu in elasticsearch.yml, not per-index.

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
            client.indices.create(index=index_name, body=index_config)

            from elasticsearch.helpers import bulk
            chunk_size = 1000
            progress_interval = max(50, n_vectors // (chunk_size * 20))  # ~20 progress lines
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
                if progress_interval and (i // chunk_size) % progress_interval == 0:
                    print(f"    Indexed {min(i + chunk_size, n_vectors):,}/{n_vectors:,} vectors")

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
        """Run kNN search over all search-param combinations and compute recall."""
        if dry_run:
            return SearchResult(
                neighbors=np.zeros((0, k), dtype=np.int64),
                distances=np.zeros((0, k), dtype=np.float32),
                search_time_ms=0,
                queries_per_second=0,
                recall=0,
                algorithm=self.algo,
                search_params=[],
                success=True,
            )

        skip_reason = self._pre_flight_check()
        if skip_reason:
            return SearchResult(
                neighbors=np.zeros((0, k), dtype=np.int64),
                distances=np.zeros((0, k), dtype=np.float32),
                search_time_ms=0,
                queries_per_second=0,
                recall=0,
                algorithm=self.algo,
                search_params=[],
                success=False,
                error_message=f"pre-flight check failed: {skip_reason}",
            )

        if not indexes:
            return SearchResult(
                neighbors=np.zeros((0, k), dtype=np.int64),
                distances=np.zeros((0, k), dtype=np.float32),
                search_time_ms=0,
                queries_per_second=0,
                recall=0,
                algorithm=self.algo,
                search_params=[],
                success=False,
                error_message="No indexes provided",
            )

        if not dataset.query_file:
            return SearchResult(
                neighbors=np.zeros((0, k), dtype=np.int64),
                distances=np.zeros((0, k), dtype=np.float32),
                search_time_ms=0,
                queries_per_second=0,
                recall=0,
                algorithm=self.algo,
                search_params=[],
                success=False,
                error_message="query_file is required",
            )

        query_path = Path(dataset.query_file)
        if not query_path.exists():
            return SearchResult(
                neighbors=np.zeros((0, k), dtype=np.int64),
                distances=np.zeros((0, k), dtype=np.float32),
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

            groundtruth = dataset.groundtruth_neighbors
            if groundtruth is None and dataset.groundtruth_neighbors_file:
                gt_path = Path(dataset.groundtruth_neighbors_file)
                if gt_path.exists():
                    groundtruth = _load_ibin(gt_path)

            index_name = self.config.get("index_name", "cuvs_bench_vectors")
            index_cfg = indexes[0]
            search_params_list = index_cfg.search_params or [{}]

            per_param_results: List[Dict[str, Any]] = []
            last_neighbors = np.full((n_queries, k), -1, dtype=np.int64)
            last_distances = np.zeros((n_queries, k), dtype=np.float32)

            for sp in search_params_list:
                num_candidates = sp.get("num_candidates", _DEFAULT_NUM_CANDIDATES)
                vector_field = sp.get("vector_field", _DEFAULT_VECTOR_FIELD)

                neighbors = np.full((n_queries, k), -1, dtype=np.int64)
                distances = np.zeros((n_queries, k), dtype=np.float32)
                latencies: List[float] = []

                t0 = time.perf_counter()
                for i, qv in enumerate(query_vectors):
                    body = {
                        "knn": {
                            "field": vector_field,
                            "query_vector": qv.tolist(),
                            "k": k,
                            "num_candidates": num_candidates,
                        }
                    }
                    t_q = time.perf_counter()
                    resp = self._get_client().search(index=index_name, body=body, size=k)
                    latencies.append((time.perf_counter() - t_q) * 1000)
                    hits = resp.get("hits", {}).get("hits", [])
                    for j, hit in enumerate(hits[:k]):
                        neighbors[i, j] = int(hit["_id"])
                        distances[i, j] = float(hit["_score"])

                elapsed_ms = (time.perf_counter() - t0) * 1000
                qps = n_queries / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0

                recall = 0.0
                if groundtruth is not None:
                    gt_k = min(k, groundtruth.shape[1])
                    n_correct = sum(
                        len(set(neighbors[i, :k].tolist()) & set(groundtruth[i, :gt_k].tolist()))
                        for i in range(n_queries)
                    )
                    recall = n_correct / (n_queries * gt_k) if gt_k > 0 else 0.0

                per_param_results.append({
                    "search_params": sp,
                    "search_time_ms": elapsed_ms,
                    "queries_per_second": qps,
                    "recall": recall,
                    "p50_ms": float(np.percentile(latencies, 50)),
                    "p95_ms": float(np.percentile(latencies, 95)),
                    "p99_ms": float(np.percentile(latencies, 99)),
                })
                last_neighbors = neighbors
                last_distances = distances

            avg_recall = float(np.mean([r["recall"] for r in per_param_results]))
            avg_qps = float(np.mean([r["queries_per_second"] for r in per_param_results]))
            total_ms = float(sum(r["search_time_ms"] for r in per_param_results))

            return SearchResult(
                neighbors=last_neighbors,
                distances=last_distances,
                search_time_ms=total_ms,
                queries_per_second=avg_qps,
                recall=avg_recall,
                algorithm=self.algo,
                search_params=search_params_list,
                latency_percentiles={
                    "p50_ms": per_param_results[-1]["p50_ms"],
                    "p95_ms": per_param_results[-1]["p95_ms"],
                    "p99_ms": per_param_results[-1]["p99_ms"],
                },
                metadata={"per_search_param_results": per_param_results},
                success=True,
            )
        except Exception as e:
            return SearchResult(
                neighbors=np.zeros((0, k), dtype=np.int64),
                distances=np.zeros((0, k), dtype=np.float32),
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
    """Get the config directory for elastic.yaml."""
    return os.path.join(os.path.dirname(__file__), "../config")


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
        username = kwargs.pop("username", None)
        password = kwargs.pop("password", None)
        scheme = kwargs.pop("scheme", "http")
        if basic_auth is None and username and password:
            basic_auth = (username, password)

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

        def _resolve(rel: Optional[str]) -> Optional[str]:
            if rel and not os.path.isabs(rel):
                return os.path.join(dataset_path, rel)
            return rel

        dataset_config = DatasetConfig(
            name=dataset_conf["name"],
            base_file=_resolve(dataset_conf.get("base_file")),
            query_file=_resolve(dataset_conf.get("query_file")),
            groundtruth_neighbors_file=_resolve(
                dataset_conf.get("groundtruth_neighbors_file")
            ),
            distance=dataset_conf.get("distance", "euclidean"),
            dims=dataset_conf.get("dims"),
            subset_size=subset_size,
        )

        if tune_mode and tune_build_params is not None and tune_search_params is not None:
            algo_name = algorithms or "elastic_hnsw"
            build_param = dict(tune_build_params)
            if "type" not in build_param and algo_name.startswith("elastic_"):
                build_param["type"] = algo_name.replace("elastic_", "", 1)
            name_parts = [f"{k}{v}" for k, v in build_param.items() if k in ("m", "ef_construction")]
            index_label = "_".join([f"{algo_name}_tune"] + name_parts) if name_parts else f"{algo_name}_tune"
            es_index_name = index_label.lower().replace(".", "_")
            index_config = IndexConfig(
                name=index_label,
                algo=algo_name,
                build_param=build_param,
                search_params=[tune_search_params],
                file="",
            )
            config = BenchmarkConfig(
                indexes=[index_config],
                backend_config={
                    "name": index_label,
                    "host": host,
                    "port": port,
                    "scheme": scheme,
                    "index_name": es_index_name,
                    "basic_auth": basic_auth,

                    **build_param,
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

        search_params_list = [
            dict(zip(search_keys, svals)) for svals in search_combos
        ]

        benchmark_configs = []
        for bvals in build_combos:
            bdict = dict(zip(build_keys, bvals))
            algo_name = f"elastic_{bdict.get('type', 'hnsw')}"

            # Derive a unique, human-readable index name from build params
            prefix = f"{algo_name}_{group_name}" if group_name != "base" else algo_name
            name_parts = [f"{k}{v}" for k, v in bdict.items() if k in ("m", "ef_construction")]
            index_label = "_".join([prefix] + name_parts) if name_parts else prefix
            es_index_name = index_label.lower().replace(".", "_")

            index_config = IndexConfig(
                name=index_label,
                algo=algo_name,
                build_param=bdict,
                search_params=search_params_list,
                file="",
            )
            config = BenchmarkConfig(
                indexes=[index_config],
                backend_config={
                    "name": index_label,
                    "host": host,
                    "port": port,
                    "scheme": scheme,
                    "index_name": es_index_name,
                    "basic_auth": basic_auth,

                    **bdict,
                },
            )
            benchmark_configs.append(config)

        return dataset_config, benchmark_configs


def register() -> None:
    """Register Elasticsearch backend and config loader (idempotent)."""
    from cuvs_bench.backends.registry import (
        _CONFIG_LOADER_REGISTRY,
        get_registry,
    )

    reg = get_registry()
    if not reg.is_registered("elastic"):
        register_backend("elastic", ElasticBackend)
    if "elastic" not in _CONFIG_LOADER_REGISTRY:
        register_config_loader("elastic", ElasticConfigLoader)
