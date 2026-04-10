#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""
OpenSearch benchmark backend for cuvs-bench supporting nmslib (HNSW), faiss,
and lucene engines for approximate nearest-neighbor search.

It also supports the remote index build service (OpenSearch 3.0+),
which offloads Faiss HNSW graph construction to a GPU-accelerated external service.
https://docs.opensearch.org/latest/vector-search/remote-index-build/
"""

import itertools
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from .base import BenchmarkBackend, BuildResult, Dataset, SearchResult
from ..orchestrator.config_loaders import (
    ConfigLoader,
    DatasetConfig,
    BenchmarkConfig,
    IndexConfig,
)


def _load_vectors(
    path: str, subset_size: Optional[int] = None
) -> np.ndarray:
    """Read a binary vector file (.fbin, .ibin, .u8bin, ...).

    The file format is: 4-byte uint32 n_rows, 4-byte uint32 n_cols,
    followed by n_rows x n_cols elements of the matching dtype.
    """
    _DTYPE_FOR_EXT = {
        ".fbin": np.float32,
        ".f16bin": np.float16,
        ".u8bin": np.uint8,
        ".i8bin": np.int8,
        ".ibin": np.int32,
    }

    ext = os.path.splitext(path)[1].lower()
    dtype = _DTYPE_FOR_EXT.get(ext, np.float32)
    with open(path, "rb") as f:
        n_rows = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
        n_cols = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
        if subset_size is not None:
            n_rows = min(n_rows, subset_size)
        raw = f.read(n_rows * n_cols * np.dtype(dtype).itemsize)
        data = np.frombuffer(raw, dtype=dtype)
    return data.reshape(n_rows, n_cols)


def _load_yaml(path: str) -> Any:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _get_dataset_conf(dataset: str, all_confs: list) -> dict:
    for d in all_confs:
        if dataset == d["name"]:
            return d
    raise ValueError(
        f"Could not find a dataset configuration for {dataset!r} in datasets.yaml"
    )


def _gather_algo_configs(
    config_path: str, algorithm_configuration: Optional[str]
) -> List[str]:
    """Return file paths of opensearch-prefixed algo YAML files."""
    algos_dir = os.path.join(config_path, "algos")
    files: List[str] = [
        os.path.join(algos_dir, f)
        for f in os.listdir(algos_dir)
        if f.startswith("opensearch") and f.endswith(".yaml")
    ]
    if algorithm_configuration:
        if os.path.isdir(algorithm_configuration):
            files += [
                os.path.join(algorithm_configuration, f)
                for f in os.listdir(algorithm_configuration)
                if f.startswith("opensearch") and f.endswith(".yaml")
            ]
        elif os.path.isfile(algorithm_configuration):
            files.append(algorithm_configuration)
    return files


class OpenSearchConfigLoader(ConfigLoader):
    """
    Configuration loader for the OpenSearch backend.

    Reads the shared ``datasets.yaml`` and opensearch-prefixed algorithm YAML
    files from the standard config directory. Expands the Cartesian product of
    build-parameter lists and returns one :class:`BenchmarkConfig` per
    build-parameter combination (with one :class:`IndexConfig` each carrying
    the full list of search-parameter combinations).

    Registration
    ------------
        from cuvs_bench.orchestrator import register_config_loader
        register_config_loader("opensearch", OpenSearchConfigLoader)
    """

    _DEFAULT_CONFIG_PATH: str = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../config"
    )

    @property
    def backend_type(self) -> str:
        return "opensearch"

    def load(
        self,
        dataset: str,
        dataset_path: str,
        algorithms: Optional[str] = None,
        count: int = 10,
        batch_size: int = 10000,
        **kwargs,
    ) -> Tuple[DatasetConfig, List[BenchmarkConfig]]:
        """
        Load OpenSearch benchmark configurations.

        Parameters
        ----------
        dataset : str
            Dataset name; must appear in ``datasets.yaml``.
        dataset_path : str
            Root directory where dataset files live.
        algorithms : Optional[str]
            Comma-separated algorithm names to run (e.g. ``"opensearch_hnsw"``).
            If *None*, all opensearch-prefixed algorithms are included.
        count : int
            Number of neighbors *k* (informational for this loader).
        batch_size : int
            Query batch size (informational for this loader).
        **kwargs
            Recognized extra keys:

            - ``dataset_configuration`` – path to a custom ``datasets.yaml``
            - ``algorithm_configuration`` – path to an extra algo config dir/file
            - ``groups`` – comma-separated group names to restrict to
            - ``subset_size`` – limit dataset to first *N* vectors
            - ``host`` – OpenSearch host (default: ``"localhost"``)
            - ``port`` – OpenSearch port (default: ``9200``)
            - ``username`` – HTTP auth username (default: ``"admin"``)
            - ``password`` – HTTP auth password (default: ``"admin"``)
            - ``use_ssl`` – whether to use HTTPS (default: ``False``)
            - ``verify_certs`` – verify SSL certificates (default: ``False``)
            - ``bulk_batch_size`` – vectors per bulk request (default: ``500``)
            - ``remote_index_build`` – enable GPU remote index build (default: ``False``)
            - ``remote_build_size_min`` – minimum segment size to trigger GPU build (default: ``"1kb"``)
            - ``remote_build_s3_endpoint`` – S3 endpoint URL
            - ``remote_build_s3_bucket`` – bucket name (default: ``"opensearch-vectors"``)
            - ``remote_build_s3_prefix`` – key prefix for ``.faiss`` polling (default: ``"knn-indexes/"``)
            - ``remote_build_s3_access_key`` – S3 access key
            - ``remote_build_s3_secret_key`` – S3 secret key
            - ``remote_build_timeout`` – GPU build timeout in seconds (default: ``600``)

        Returns
        -------
        Tuple[DatasetConfig, List[BenchmarkConfig]]
        """
        config_path = self._DEFAULT_CONFIG_PATH

        ds_yaml = kwargs.get("dataset_configuration") or os.path.join(
            config_path, "datasets", "datasets.yaml"
        )
        all_ds = _load_yaml(ds_yaml)
        ds_conf = _get_dataset_conf(dataset, all_ds)

        def _resolve(rel: Optional[str]) -> Optional[str]:
            if rel and not os.path.isabs(rel):
                return os.path.join(dataset_path, rel)
            return rel

        dataset_config = DatasetConfig(
            name=ds_conf["name"],
            base_file=_resolve(ds_conf.get("base_file")),
            query_file=_resolve(ds_conf.get("query_file")),
            groundtruth_neighbors_file=_resolve(
                ds_conf.get("groundtruth_neighbors_file")
            ),
            distance=ds_conf.get("distance", "euclidean"),
            dims=ds_conf.get("dims"),
            subset_size=kwargs.get("subset_size"),
        )

        algo_files = _gather_algo_configs(
            config_path, kwargs.get("algorithm_configuration")
        )

        allowed_algos = (
            [a.strip() for a in algorithms.split(",")] if algorithms else None
        )
        allowed_groups = (
            [g.strip() for g in kwargs["groups"].split(",")]
            if kwargs.get("groups")
            else None
        )

        host = kwargs.get("host", "localhost")
        port = kwargs.get("port", 9200)

        # Connection and remote-build kwargs forwarded verbatim to backend_config
        _conn_keys = (
            "username",
            "password",
            "use_ssl",
            "verify_certs",
            "bulk_batch_size",
            # Remote Index Build (OpenSearch 3.0+, faiss engine only)
            "remote_index_build",
            "remote_build_size_min",
            "remote_build_s3_endpoint",
            "remote_build_s3_bucket",
            "remote_build_s3_prefix",
            "remote_build_s3_access_key",
            "remote_build_s3_secret_key",
            "remote_build_timeout",
        )
        conn_kwargs = {k: kwargs[k] for k in _conn_keys if k in kwargs}

        benchmark_configs: List[BenchmarkConfig] = []

        for algo_file in algo_files:
            algo_yaml = _load_yaml(algo_file)
            algo_name = algo_yaml.get("name", "")
            if allowed_algos and algo_name not in allowed_algos:
                continue

            groups: Dict[str, Any] = algo_yaml.get("groups", {})
            if allowed_groups:
                groups = {k: v for k, v in groups.items() if k in allowed_groups}

            for group_name, group_conf in groups.items():
                build_spec: Dict[str, List] = group_conf.get("build", {})
                search_spec: Dict[str, List] = group_conf.get("search", {})

                build_keys = list(build_spec.keys())
                build_combos = (
                    list(itertools.product(*build_spec.values()))
                    if build_spec
                    else [()]
                )

                search_keys = list(search_spec.keys())
                search_combos = (
                    list(itertools.product(*search_spec.values()))
                    if search_spec
                    else [()]
                )
                search_params_list = [
                    dict(zip(search_keys, vals)) for vals in search_combos
                ]

                for build_vals in build_combos:
                    build_param = dict(zip(build_keys, build_vals))

                    # Human-readable index label
                    prefix = (
                        f"{algo_name}_{group_name}"
                        if group_name != "base"
                        else algo_name
                    )
                    parts = [prefix] + [f"{k}{v}" for k, v in build_param.items()]
                    index_label = ".".join(parts)

                    # OpenSearch index names must be lowercase with no dots
                    os_index_name = index_label.replace(".", "_").lower()
                    index_file = os.path.join(
                        dataset_path, dataset, "index", index_label
                    )

                    index_cfg = IndexConfig(
                        name=index_label,
                        algo=algo_name,
                        build_param=build_param,
                        search_params=search_params_list,
                        file=index_file,
                    )

                    engine = "nmslib"
                    if "faiss" in algo_name:
                        engine = "faiss"
                    elif "lucene" in algo_name:
                        engine = "lucene"

                    backend_cfg: Dict[str, Any] = {
                        "name": index_label,
                        "host": host,
                        "port": port,
                        "index_name": os_index_name,
                        "engine": engine,
                        "algo": algo_name,
                        "requires_network": True,
                        **conn_kwargs,
                    }

                    benchmark_configs.append(
                        BenchmarkConfig(
                            indexes=[index_cfg],
                            backend_config=backend_cfg,
                        )
                    )

        return dataset_config, benchmark_configs


# Mapping from cuvs-bench distance metric names to OpenSearch space_type
_DISTANCE_TO_SPACE_TYPE: Dict[str, str] = {
    "euclidean": "l2",
    "l2": "l2",
    "inner_product": "innerproduct",
    "innerproduct": "innerproduct",
    "cosine": "cosinesimil",
    "cosinesimil": "cosinesimil",
    "angular": "cosinesimil",
}


class OpenSearchBackend(BenchmarkBackend):
    """
    Benchmark backend for OpenSearch's k-NN plugin.

    Supports the nmslib (HNSW), faiss (HNSW / IVF), and lucene (HNSW)
    engines. Vectors are bulk-indexed as ``knn_vector`` fields and retrieved
    via the standard ``knn`` query type.

    Requires ``opensearch-py`` Python package.

    Parameters
    ----------
    config : Dict[str, Any]
        Backend configuration produced by :class:`OpenSearchConfigLoader`.
        Recognized keys:

        Required:
        - ``name`` – index label (e.g. ``"opensearch_hnsw.m16.ef_construction100"``)
        - ``index_name`` – OpenSearch index name (lowercase, no dots)
        - ``engine`` – ``"nmslib"``, ``"faiss"``, or ``"lucene"``
        - ``algo`` – algorithm name (e.g. ``"opensearch_hnsw"``)

        Optional:
        - ``host`` – hostname (default: ``"localhost"``)
        - ``port`` – port (default: ``9200``)
        - ``username`` – HTTP basic auth user (default: ``"admin"``)
        - ``password`` – HTTP basic auth password (default: ``"admin"``)
        - ``use_ssl`` – use HTTPS (default: ``False``)
        - ``verify_certs`` – verify SSL certs (default: ``False``)
        - ``bulk_batch_size`` – vectors per bulk request (default: ``500``)
        - ``requires_network`` – trigger network pre-flight check (default: ``True``)
        - ``remote_index_build`` – set ``index.knn.remote_index_build.enabled=true``
          on the index at creation time, opting it into the GPU build path (default: ``False``).
        - ``remote_build_size_min`` – minimum segment size to trigger GPU build, e.g. ``"1kb"``
          (default: ``"1kb"`` when ``remote_index_build=True``)
        - ``remote_build_s3_endpoint`` – S3 endpoint URL for build polling
        - ``remote_build_s3_bucket`` – bucket name (default: ``"opensearch-vectors"``)
        - ``remote_build_s3_prefix`` – key prefix to scan for ``.faiss`` files (default: ``"knn-indexes/"``)
        - ``remote_build_s3_access_key`` – S3 access key
        - ``remote_build_s3_secret_key`` – S3 secret key
        - ``remote_build_timeout`` – seconds to wait for GPU build (default: ``600``)

    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.__client = None

    @property
    def algo(self) -> str:
        return self.config.get("algo", "opensearch")

    @property
    def _client(self):
        if self.__client is None:
            try:
                from opensearchpy import OpenSearch
            except ImportError as e:
                raise ImportError(
                    "`opensearch-py` is required for the OpenSearch backend in cuvs-bench.\n\n"
                    "Install it with: `pip install opensearch-py`"
                ) from e

            host = self.config.get("host", "localhost")
            port = self.config.get("port", 9200)
            username = self.config.get("username", "admin")
            password = self.config.get("password", "admin")
            use_ssl = self.config.get("use_ssl", False)
            verify_certs = self.config.get("verify_certs", False)

            self.__client = OpenSearch(
                hosts=[{"host": host, "port": port}],
                http_auth=(username, password) if username else None,
                use_ssl=use_ssl,
                verify_certs=verify_certs,
            )
        return self.__client

    def _check_network_available(self) -> bool:
        try:
            self._client.cluster.health(request_timeout=5)
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def initialize(self) -> None:
        """Eagerly open the connection to OpenSearch."""
        _ = self._client

    def cleanup(self) -> None:
        """Close the OpenSearch connection."""
        if self.__client is not None:
            self.__client.close()
            self.__client = None

    def _build_index_mapping(
        self,
        dims: int,
        engine: str,
        space_type: str,
        build_param: Dict[str, Any],
        remote_index_build: bool = False,
        remote_build_size_min: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Construct the OpenSearch index mapping dict for k-NN.

        The mapping enables the k-NN plugin and configures the vector field
        with the chosen engine and method parameters.

        When ``remote_index_build=True``, ``index.knn.remote_index_build.enabled``
        is set to ``true`` in the index settings, opting qualifying segments into
        the GPU build path.  ``remote_build_size_min`` sets
        ``index.knn.remote_index_build.size.min`` to control the minimum segment
        size that triggers the GPU build; the default ``"1kb"`` ensures any
        non-trivial segment is built remotely.  The cluster-level infrastructure
        is assumed to be pre-configured externally.
        """
        m = build_param.get("m", 16)
        ef_construction = build_param.get("ef_construction", 100)

        if engine in ("nmslib", "lucene"):
            method_name = "hnsw"
            method_params: Dict[str, Any] = {
                "m": m,
                "ef_construction": ef_construction,
            }
        elif engine == "faiss":
            faiss_method = build_param.get("faiss_method", "hnsw")
            method_name = faiss_method
            if faiss_method == "hnsw":
                method_params = {
                    "m": m,
                    "ef_construction": ef_construction,
                }
            elif faiss_method == "ivf":
                method_params = {"nlist": build_param.get("nlist", 4)}
            else:
                raise ValueError(
                    f"Unsupported faiss_method {faiss_method!r}. "
                    "Use 'hnsw' or 'ivf'."
                )
        else:
            raise ValueError(
                f"Unknown OpenSearch k-NN engine {engine!r}. "
                "Use 'nmslib', 'faiss', or 'lucene'."
            )

        index_settings = {
            "knn": True,
            "number_of_shards": build_param.get("number_of_shards", 1),
            "number_of_replicas": build_param.get("number_of_replicas", 0),
        }
        if remote_index_build:
            if engine != "faiss":
                raise ValueError(
                    "Remote Index Build only supports the faiss engine "
                    f"(got engine={engine!r}). "
                    "Use algorithms='opensearch_faiss_hnsw'."
                )
            index_settings["knn.remote_index_build.enabled"] = True
            index_settings["knn.remote_index_build.size.min"] = (
                remote_build_size_min or "1kb"
            )

        return {
            "settings": {"index": index_settings},
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": dims,
                        "method": {
                            "name": method_name,
                            "space_type": space_type,
                            "engine": engine,
                            "parameters": method_params,
                        },
                    }
                }
            },
        }

    def _bulk_index(
        self,
        index_name: str,
        vectors: np.ndarray,
        bulk_batch_size: int,
    ) -> None:
        """
        Bulk-index vectors into index_name using the helpers API.

        Vectors are stored under the ``"vector"`` field with their integer
        row index as the document ``_id`` so they can be mapped back to
        ground-truth neighbor lists.
        """
        from opensearchpy.helpers import streaming_bulk

        def _doc_generator():
            for i, vec in enumerate(vectors):
                yield {
                    "_index": index_name,
                    "_id": str(i),
                    "vector": vec.tolist(),
                }

        total = vectors.shape[0]
        indexed = 0
        for ok, info in streaming_bulk(
            self._client,
            _doc_generator(),
            chunk_size=bulk_batch_size,
            request_timeout=120,
        ):
            if not ok:
                raise RuntimeError(f"Failed to index document: {info}")
            indexed += 1
            if indexed % max(bulk_batch_size, 1000) == 0:
                print(f"  Indexed {indexed} / {total} vectors")
        print(f"  Indexed all {total} vectors")

    def _wait_for_remote_build(
        self,
        expected_count: int = 1,
        timeout: int = 600,
        poll_interval: int = 5,
    ) -> None:
        """
        Poll S3 for .faiss files confirming the GPU remote build completed.

        Raises ``TimeoutError`` if the expected number of files does not appear
        within *timeout* seconds.
        """
        try:
            import boto3
            from botocore.config import Config as BotocoreConfig
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for remote index build polling. "
                "Install it with:  pip install boto3"
            ) from exc

        s3_endpoint = self.config.get("remote_build_s3_endpoint")
        s3_bucket = self.config.get("remote_build_s3_bucket", "opensearch-vectors")
        s3_prefix = self.config.get("remote_build_s3_prefix", "knn-indexes/")
        s3_access_key = self.config.get("remote_build_s3_access_key")
        s3_secret_key = self.config.get("remote_build_s3_secret_key")

        s3 = boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
            region_name="us-east-1",
            config=BotocoreConfig(signature_version="s3v4"),
        )

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
                faiss_files = [
                    obj["Key"]
                    for obj in resp.get("Contents", [])
                    if obj["Key"].endswith(".faiss")
                ]
                if len(faiss_files) >= expected_count:
                    return
            except Exception as exc:
                print(f"  S3 poll error: {exc}")
            time.sleep(poll_interval)

        raise TimeoutError(
            f"GPU build not confirmed after {timeout}s: "
            f"expected {expected_count} .faiss file(s) in s3://{s3_bucket}/{s3_prefix}"
        )

    def build(
        self,
        dataset: Dataset,
        indexes: List[IndexConfig],
        force: bool = False,
        dry_run: bool = False,
    ) -> BuildResult:
        """
        Build an OpenSearch k-NN index from the dataset's base vectors.

        Creates an index with k-NN plugin mapping and bulk-indexes all base
        vectors.  If the index already exists and ``force=False`` the build
        is skipped.

        When ``remote_index_build=True`` the build time includes the full GPU
        build flow: ingest → force merge → wait for ``.faiss`` files in S3.

        Parameters
        ----------
        dataset : Dataset
            Must have either non-empty ``base_vectors`` or a valid
            ``base_file`` path.
        indexes : List[IndexConfig]
            The first element provides the build parameters.
        force : bool
            Delete and recreate the index if it already exists.
        dry_run : bool
            Log what would happen without making any changes.

        Returns
        -------
        BuildResult
        """
        skip = self._pre_flight_check()
        if skip:
            return BuildResult(
                index_path="",
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params={},
                success=False,
                error_message=f"pre-flight check failed: {skip}",
            )

        if not indexes:
            return BuildResult(
                index_path="",
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params={},
                success=False,
                error_message="No indexes provided",
            )

        index_cfg = indexes[0]
        build_param = index_cfg.build_param
        index_name = self.config.get(
            "index_name", index_cfg.name.replace(".", "_").lower()
        )
        engine = self.config.get("engine", "nmslib")
        bulk_batch_size = int(self.config.get("bulk_batch_size", 500))
        remote_index_build = bool(self.config.get("remote_index_build", False))
        remote_build_size_min = self.config.get("remote_build_size_min")

        if dry_run:
            print(f"[dry_run] Would build OpenSearch index '{index_name}' "
                  f"(engine={engine}, remote_index_build={remote_index_build}, build_param={build_param})")

            return BuildResult(
                index_path=index_name,
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params=build_param,
                success=True,
            )


        # Handle existing index
        if self._client.indices.exists(index=index_name):
            if force:
                self._client.indices.delete(index=index_name)
            else:
                return BuildResult(
                    index_path=index_name,
                    build_time_seconds=0.0,
                    index_size_bytes=0,
                    algorithm=self.algo,
                    build_params=build_param,
                    success=True,
                )

        # Load base vectors (may be empty if only file path is provided)
        base_vectors = dataset.base_vectors
        subset_size = dataset.metadata.get("subset_size")
        if base_vectors.size == 0 and dataset.base_file:
            base_vectors = _load_vectors(dataset.base_file, subset_size)

        if base_vectors.size == 0:
            return BuildResult(
                index_path="",
                build_time_seconds=0.0,
                index_size_bytes=0,
                algorithm=self.algo,
                build_params=build_param,
                success=False,
                error_message=(
                    "No base vectors available. Provide dataset.base_vectors "
                    "or a valid dataset.base_file path."
                ),
            )

        dims = base_vectors.shape[1]
        space_type = _DISTANCE_TO_SPACE_TYPE.get(
            dataset.distance_metric, "l2"
        )

        # Create index
        mapping = self._build_index_mapping(
            dims, engine, space_type, build_param, remote_index_build, remote_build_size_min
        )
        self._client.indices.create(index=index_name, body=mapping)

        # Bulk index, then trigger and await the GPU build (if remote)
        t0 = time.perf_counter()
        self._bulk_index(index_name, base_vectors, bulk_batch_size)
        self._client.indices.refresh(index=index_name, request_timeout=120)
        if remote_index_build:
            # Force-merge `index_name` to one segment, initiating the remote GPU build.
            self._client.indices.forcemerge(
                index=index_name, max_num_segments=1, request_timeout=300
            )
            self._wait_for_remote_build(
                expected_count=1,
                timeout=int(self.config.get("remote_build_timeout", 600)),
            )
        build_time = time.perf_counter() - t0

        # Index size
        stats = self._client.indices.stats(index=index_name)
        index_size_bytes = int(
            stats["_all"]["total"]["store"]["size_in_bytes"]
        )

        return BuildResult(
            index_path=index_name,
            build_time_seconds=build_time,
            index_size_bytes=index_size_bytes,
            algorithm=self.algo,
            build_params=build_param,
            metadata={
                "engine": engine,
                "space_type": space_type,
                "remote_index_build": remote_index_build,
            },
            success=True,
        )

    def search(
        self,
        dataset: Dataset,
        indexes: List[IndexConfig],
        k: int,
        batch_size: int = 10000,
        mode: str = "latency",
        force: bool = False,
        search_threads: Optional[int] = None,
        dry_run: bool = False,
    ) -> SearchResult:
        """
        Search the OpenSearch k-NN index for nearest neighbors.

        Iterates over every search-parameter combination defined in the index
        config, updating the index-level ``ef_search`` setting between runs.
        Metrics (recall, QPS, latency) are collected per parameter set and
        stored in ``SearchResult.metadata["per_search_param_results"]``.

        The *neighbors* and *distances* arrays in the returned result reflect
        the **last** search-parameter combination (highest ef_search by
        convention), while *recall* and *queries_per_second* are the averages
        across all parameter combinations.

        Parameters
        ----------
        dataset : Dataset
            Must have either non-empty ``query_vectors`` or a valid
            ``query_file`` path.  Ground truth is loaded from
            ``groundtruth_neighbors_file`` if available.
        indexes : List[IndexConfig]
            The first element is used; its ``search_params`` list defines the
            ef_search values to sweep.
        k : int
            Number of neighbors to retrieve per query.
        batch_size : int
            Unused; included for interface compatibility.
        mode : str
            ``"latency"`` or ``"throughput"``; informational for this backend.
        force : bool
            Unused; included for interface compatibility.
        search_threads : Optional[int]
            Unused; OpenSearch manages its own threading.
        dry_run : bool
            Log what would happen without making any OpenSearch calls.

        Returns
        -------
        SearchResult
        """
        skip = self._pre_flight_check()
        if skip:
            return SearchResult(
                neighbors=np.zeros((0, k), dtype=np.int64),
                distances=np.zeros((0, k), dtype=np.float32),
                search_time_ms=0.0,
                queries_per_second=0.0,
                recall=0.0,
                algorithm=self.algo,
                search_params=[],
                success=False,
                error_message=f"pre-flight check failed: {skip}",
            )

        if not indexes:
            return SearchResult(
                neighbors=np.zeros((0, k), dtype=np.int64),
                distances=np.zeros((0, k), dtype=np.float32),
                search_time_ms=0.0,
                queries_per_second=0.0,
                recall=0.0,
                algorithm=self.algo,
                search_params=[],
                success=False,
                error_message="No indexes provided",
            )

        index_cfg = indexes[0]
        index_name = self.config.get(
            "index_name", index_cfg.name.replace(".", "_").lower()
        )
        engine = self.config.get("engine", "nmslib")
        search_params_list = index_cfg.search_params or [{}]

        if dry_run:
            print(f"[dry_run] Would search OpenSearch index '{index_name}' with {len(search_params_list)} param set(s) (k={k})")

            return SearchResult(
                neighbors=np.zeros((0, k), dtype=np.int64),
                distances=np.zeros((0, k), dtype=np.float32),
                search_time_ms=0.0,
                queries_per_second=0.0,
                recall=0.0,
                algorithm=self.algo,
                search_params=search_params_list,
                success=True,
            )

        # Load query vectors and ground truth from files if not already loaded
        query_vectors = dataset.query_vectors
        if query_vectors.size == 0 and dataset.query_file:
            query_vectors = _load_vectors(dataset.query_file)

        if query_vectors.size == 0:
            return SearchResult(
                neighbors=np.zeros((0, k), dtype=np.int64),
                distances=np.zeros((0, k), dtype=np.float32),
                search_time_ms=0.0,
                queries_per_second=0.0,
                recall=0.0,
                algorithm=self.algo,
                search_params=search_params_list,
                success=False,
                error_message=(
                    "No query vectors available. Provide dataset.query_vectors "
                    "or a valid dataset.query_file path."
                ),
            )

        groundtruth = dataset.groundtruth_neighbors
        if groundtruth is None and dataset.groundtruth_neighbors_file:
            groundtruth = _load_vectors(dataset.groundtruth_neighbors_file)

        n_queries = query_vectors.shape[0]

        # Run search for each search-parameter combination
        per_param_results: List[Dict[str, Any]] = []
        last_neighbors = np.full((n_queries, k), -1, dtype=np.int64)
        last_distances = np.zeros((n_queries, k), dtype=np.float32)

        for sp in search_params_list:
            ef_search = sp.get("ef_search", 100)

            if engine in ("nmslib", "faiss"):
                self._client.indices.put_settings(
                    index=index_name,
                    body={"index.knn.algo_param.ef_search": ef_search},
                )

            neighbors = np.full((n_queries, k), -1, dtype=np.int64)
            distances = np.zeros((n_queries, k), dtype=np.float32)

            t0 = time.perf_counter()
            for i, q_vec in enumerate(query_vectors):
                body: Dict[str, Any] = {
                    "size": k,
                    "query": {
                        "knn": {
                            "vector": {
                                "vector": q_vec.tolist(),
                                "k": k,
                            }
                        }
                    },
                }
                resp = self._client.search(index=index_name, body=body)
                hits = resp["hits"]["hits"]
                for j, hit in enumerate(hits[:k]):
                    neighbors[i, j] = int(hit["_id"])
                    distances[i, j] = float(hit["_score"])

            elapsed = time.perf_counter() - t0
            qps = n_queries / elapsed if elapsed > 0 else 0.0

            recall = 0.0
            if groundtruth is not None:
                gt_k = min(k, groundtruth.shape[1])
                gt = groundtruth[:, :gt_k]
                n_correct = sum(
                    len(set(neighbors[i, :k].tolist()) & set(gt[i].tolist()))
                    for i in range(n_queries)
                )
                recall = n_correct / (n_queries * gt_k)

            per_param_results.append(
                {
                    "search_params": sp,
                    "search_time_ms": elapsed * 1000.0,
                    "queries_per_second": qps,
                    "recall": recall,
                }
            )
            last_neighbors = neighbors
            last_distances = distances

        # Aggregate across all search-param combinations
        avg_recall = float(
            np.mean([r["recall"] for r in per_param_results])
        )
        avg_qps = float(
            np.mean([r["queries_per_second"] for r in per_param_results])
        )
        total_search_time_ms = float(
            sum(r["search_time_ms"] for r in per_param_results)
        )

        return SearchResult(
            neighbors=last_neighbors,
            distances=last_distances,
            search_time_ms=total_search_time_ms,
            queries_per_second=avg_qps,
            recall=avg_recall,
            algorithm=self.algo,
            search_params=search_params_list,
            metadata={
                "engine": engine,
                "per_search_param_results": per_param_results,
            },
            success=True,
        )
