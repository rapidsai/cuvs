#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""
Tests for the OpenSearch benchmark backend.
"""

import os
from urllib.parse import urlparse

import numpy as np
import pytest
import requests

pytest.importorskip("opensearchpy")

from cuvs_bench.backends.base import Dataset
from cuvs_bench.backends.opensearch import OpenSearchBackend, OpenSearchConfigLoader
from cuvs_bench.orchestrator.config_loaders import IndexConfig



def _make_backend(config_overrides: dict = None) -> OpenSearchBackend:
    """Backend with no network requirement so pre-flight passes without a server."""
    config = {
        "name": "test_index",
        "index_name": "test_index",
        "engine": "faiss",
        "algo": "opensearch_faiss_hnsw",
        **(config_overrides or {}),
    }
    return OpenSearchBackend(config)


def _make_dataset(n_base: int = 10, n_queries: int = 5, dims: int = 4, k: int = 10) -> Dataset:
    rng = np.random.default_rng(0)
    base = rng.random((n_base, dims)).astype(np.float32)
    queries = rng.random((n_queries, dims)).astype(np.float32)
    dists = np.sum((queries[:, None, :] - base[None, :, :]) ** 2, axis=2)
    groundtruth = np.argsort(dists, axis=1)[:, :k].astype(np.int32)
    return Dataset(
        name="test",
        base_vectors=base,
        query_vectors=queries,
        groundtruth_neighbors=groundtruth,
    )


def _make_index_cfg(search_params: list = None) -> IndexConfig:
    return IndexConfig(
        name="test_index",
        algo="opensearch_faiss_hnsw",
        build_param={"m": 16, "ef_construction": 100},
        search_params=search_params or [{"ef_search": 50}, {"ef_search": 100}],
        file="",
    )


def _cleanup_backend(backend: OpenSearchBackend, index_name: str) -> None:
    """Delete the test index and close the client connection."""
    try:
        if backend._client.indices.exists(index=index_name):
            backend._client.indices.delete(index=index_name)
    except Exception:
        pass
    backend.cleanup()



@pytest.fixture(scope="session")
def opensearch_url():
    """Skip integration tests when no live OpenSearch node is reachable."""
    url = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
    try:
        requests.get(f"{url}/_cluster/health", timeout=2).raise_for_status()
    except Exception:
        pytest.skip("no OpenSearch node reachable")
    return url


@pytest.fixture
def live_backend(opensearch_url):
    """Backend connected to a live OpenSearch node; cleans up the test index on teardown."""
    parsed = urlparse(opensearch_url)
    index_name = "cuvs_test_index"
    backend = OpenSearchBackend({
        "name": index_name,
        "index_name": index_name,
        "engine": "faiss",
        "algo": "opensearch_faiss_hnsw",
        "host": parsed.hostname,
        "port": parsed.port or 9200,
        "use_ssl": parsed.scheme == "https",
        "verify_certs": False,
        "requires_network": True,
    })
    yield backend
    _cleanup_backend(backend, index_name)


@pytest.fixture
def config_dir(tmp_path):
    """Config directory with a minimal dataset and algo definition."""
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "datasets.yaml").write_text(
        "- name: test-ds\n  distance: euclidean\n  dims: 4\n"
    )
    (tmp_path / "algos").mkdir()
    (tmp_path / "algos" / "opensearch_faiss_hnsw.yaml").write_text(
        "name: opensearch_faiss_hnsw\n"
        "groups:\n"
        "  test:\n"
        "    build:\n"
        "      m: [16, 32]\n"
        "      ef_construction: [100, 200]\n"
        "    search:\n"
        "      ef_search: [50, 100]\n"
    )
    return tmp_path



class TestOpenSearchConfigLoader:
    def test_load_produces_correct_configs(self, config_dir):
        loader = OpenSearchConfigLoader()
        loader._DEFAULT_CONFIG_PATH = str(config_dir)

        dataset_config, benchmark_configs = loader.load(
            dataset="test-ds", dataset_path="/data", groups="test"
        )

        assert dataset_config.name == "test-ds"
        assert dataset_config.distance == "euclidean"
        # m=[16,32] x ef_construction=[100,200] = 4 build combos
        assert len(benchmark_configs) == 4
        bc = benchmark_configs[0]
        assert bc.backend_config["engine"] == "faiss"
        assert len(bc.indexes[0].search_params) == 2  # ef_search: [50, 100]

    def test_load_forwards_remote_build_kwargs(self, config_dir):
        loader = OpenSearchConfigLoader()
        loader._DEFAULT_CONFIG_PATH = str(config_dir)

        _, configs = loader.load(
            dataset="test-ds",
            dataset_path="/data",
            remote_index_build=True,
            remote_build_s3_endpoint="http://s3:9000",
            remote_build_s3_bucket="my-bucket",
            remote_build_s3_access_key="mykey",
            remote_build_s3_secret_key="mysecret",
        )

        bc = configs[0].backend_config
        assert bc["remote_index_build"] is True
        assert bc["remote_build_s3_endpoint"] == "http://s3:9000"
        assert bc["remote_build_s3_bucket"] == "my-bucket"
        assert bc["remote_build_s3_access_key"] == "mykey"
        assert bc["remote_build_s3_secret_key"] == "mysecret"



class TestOpenSearchBackend:
    def test_build_dry_run(self):
        result = _make_backend().build(_make_dataset(), [_make_index_cfg()], dry_run=True)
        assert result.success
        assert result.index_path == "test_index"

    def test_search_dry_run(self):
        result = _make_backend().search(_make_dataset(), [_make_index_cfg()], k=3, dry_run=True)
        assert result.success
        assert len(result.search_params) == 2

    def test_search_fails_without_query_vectors(self):
        dataset = Dataset(
            name="empty",
            base_vectors=np.empty((0, 4), dtype=np.float32),
            query_vectors=np.empty((0, 4), dtype=np.float32),
        )
        result = _make_backend().search(dataset, [_make_index_cfg()], k=3)
        assert not result.success
        assert "No query vectors" in result.error_message



@pytest.fixture(scope="session")
def remote_build_env(opensearch_url):
    """
    Skip remote index build tests if required env vars aren't set or services aren't reachable.

    Required environment variables:
      BUILDER_URL      URL of the remote index builder service
      S3_ENDPOINT      S3-compatible object store endpoint URL
      S3_BUCKET        Bucket name used by OpenSearch for vector staging
      S3_ACCESS_KEY    S3 access key
      S3_SECRET_KEY    S3 secret key
    """
    builder_url = os.environ.get("BUILDER_URL")
    s3_endpoint = os.environ.get("S3_ENDPOINT")
    s3_bucket = os.environ.get("S3_BUCKET")
    s3_access_key = os.environ.get("S3_ACCESS_KEY")
    s3_secret_key = os.environ.get("S3_SECRET_KEY")

    missing = [
        name for name, val in {
            "BUILDER_URL": builder_url, "S3_ENDPOINT": s3_endpoint,
            "S3_BUCKET": s3_bucket, "S3_ACCESS_KEY": s3_access_key,
            "S3_SECRET_KEY": s3_secret_key,
        }.items() if not val
    ]
    if missing:
        pytest.skip(f"remote index build tests require env vars: {', '.join(missing)}")

    try:
        requests.get(builder_url, timeout=2)
    except requests.exceptions.ConnectionError:
        pytest.skip(f"remote index builder not reachable at {builder_url}")

    try:
        requests.get(s3_endpoint, timeout=2)
    except requests.exceptions.ConnectionError:
        pytest.skip(f"S3 endpoint not reachable at {s3_endpoint}")

    # Register the S3 snapshot repo and enable remote index build
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    repo_name = "vector-repo"

    session.put(
        f"{opensearch_url}/_snapshot/{repo_name}",
        json={"type": "s3", "settings": {"bucket": s3_bucket, "base_path": "knn-indexes"}},
    ).raise_for_status()

    session.put(
        f"{opensearch_url}/_cluster/settings",
        json={"persistent": {
            "knn.remote_index_build.enabled": True,
            "knn.remote_index_build.repository": repo_name,
            "knn.remote_index_build.service.endpoint": builder_url,
        }},
    ).raise_for_status()

    return {
        "s3_endpoint": s3_endpoint,
        "s3_bucket": s3_bucket,
        "s3_access_key": s3_access_key,
        "s3_secret_key": s3_secret_key,
    }


@pytest.fixture
def live_remote_build_backend(opensearch_url, remote_build_env):
    """Backend with remote_index_build=True pointing at a configured remote build stack."""
    parsed = urlparse(opensearch_url)
    index_name = "cuvs_test_remote_index"
    backend = OpenSearchBackend({
        "name": index_name,
        "index_name": index_name,
        "engine": "faiss",
        "algo": "opensearch_faiss_hnsw",
        "host": parsed.hostname,
        "port": parsed.port or 9200,
        "use_ssl": parsed.scheme == "https",
        "verify_certs": False,
        "requires_network": True,
        "remote_index_build": True,
        "remote_build_s3_endpoint": remote_build_env["s3_endpoint"],
        "remote_build_s3_bucket": remote_build_env["s3_bucket"],
        "remote_build_s3_prefix": "knn-indexes/",
        "remote_build_s3_access_key": remote_build_env["s3_access_key"],
        "remote_build_s3_secret_key": remote_build_env["s3_secret_key"],
    })
    yield backend
    _cleanup_backend(backend, index_name)



@pytest.mark.integration
class TestOpenSearchBackendIntegration:
    def test_build_and_search(self, live_backend):
        k = 3
        dataset = _make_dataset(n_base=100, n_queries=10, dims=4, k=k)
        idx = _make_index_cfg(search_params=[{"ef_search": 50}])

        build_result = live_backend.build(dataset, [idx], force=True)
        assert build_result.success
        assert build_result.build_time_seconds > 0
        assert build_result.index_size_bytes > 0

        search_result = live_backend.search(dataset, [idx], k=k)
        assert search_result.success
        assert search_result.recall > 0
        assert search_result.queries_per_second > 0
        assert len(search_result.metadata["per_search_param_results"]) == 1



@pytest.mark.integration
class TestOpenSearchRemoteIndexBuildIntegration:
    def test_remote_build_and_search(self, live_remote_build_backend):
        k = 3
        dataset = _make_dataset(n_base=100, n_queries=10, dims=4, k=k)
        idx = _make_index_cfg(search_params=[{"ef_search": 50}])

        build_result = live_remote_build_backend.build(dataset, [idx], force=True)
        assert build_result.success
        assert build_result.build_time_seconds > 0
        assert build_result.metadata["remote_index_build"] is True

        search_result = live_remote_build_backend.search(dataset, [idx], k=k)
        assert search_result.success
        assert search_result.recall > 0
        assert search_result.queries_per_second > 0
