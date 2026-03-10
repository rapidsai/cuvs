#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Integration tests for Elasticsearch backend using testcontainers.

**Currently disabled.** The cuvs-bench elastic backend targets Elasticsearch GPU
(cuVS-accelerated vector search). A proper integration test would require:
- Elasticsearch GPU image (docker.elastic.co/elasticsearch-dev/elasticsearch-gpu:9.3.x)
- cuVS libraries from this repo (built libcuvs.so) mounted into the container
- GPU-enabled CI runner (--gpus all)
- LD_LIBRARY_PATH and volume mounts configured

The standard ES OSS image has no GPU support, so tests with use_gpu=False only
validate the HTTP API, not the GPU indexing path. This can be re-enabled later
when the above dependencies are available in CI.

Requires: pip install cuvs-bench[elastic,integration]
Requires: Docker running locally

Run with:
    pytest python/cuvs_bench/cuvs_bench/tests/integration/ -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cuvs_bench.backends.base import Dataset
from cuvs_bench.backends.registry import get_backend_class
from cuvs_bench.orchestrator.config_loaders import IndexConfig


def _write_fbin(path: Path, data: np.ndarray) -> None:
    """Write big-ann-bench fbin format."""
    with open(path, "wb") as f:
        np.array(data.shape, dtype=np.uint32).tofile(f)
        data.astype(np.float32).tofile(f)


def _write_ibin(path: Path, data: np.ndarray) -> None:
    """Write big-ann-bench ibin format."""
    with open(path, "wb") as f:
        np.array(data.shape, dtype=np.uint32).tofile(f)
        data.astype(np.int32).tofile(f)


@pytest.mark.skip(
    reason="Integration test disabled: requires ES GPU image + cuVS libs + GPU runner. "
    "See module docstring. Can be re-enabled when CI has these dependencies."
)
class TestElasticIntegration:
    """Integration tests against a real Elasticsearch container.

    Disabled until CI can provide: Elasticsearch GPU image, cuVS libraries
    (from this repo build), and GPU-enabled runner. Standard ES OSS does not
    exercise the GPU vector search path this backend targets.
    """

    def test_elastic_build_and_search_e2e(self, elasticsearch_container):
        """Build index and run search against Elasticsearch container."""
        es = elasticsearch_container
        cls = get_backend_class("elastic")
        backend = cls(
            config={
                "name": "integration_test",
                "host": es["host"],
                "port": es["port"],
                "index_name": "cuvs_bench_integration_test",
                "use_gpu": False,  # Standard ES OSS has no GPU
            }
        )
        try:
            # Create temp dir with fbin/ibin files
            with tempfile.TemporaryDirectory() as tmpdir:
                base_path = Path(tmpdir) / "base.fbin"
                query_path = Path(tmpdir) / "query.fbin"
                gt_path = Path(tmpdir) / "groundtruth_neighbors.ibin"

                n_base, dims = 200, 32
                n_queries, k = 20, 10

                base = np.random.rand(n_base, dims).astype(np.float32)
                queries = np.random.rand(n_queries, dims).astype(np.float32)
                # Ground truth: first k neighbors per query (dummy IDs)
                gt = np.random.randint(0, n_base, size=(n_queries, k), dtype=np.int32)

                _write_fbin(base_path, base)
                _write_fbin(query_path, queries)
                _write_ibin(gt_path, gt)

                backend.config["data_prefix"] = str(tmpdir)

                dataset = Dataset(
                    name="integration_test",
                    base_vectors=base,
                    query_vectors=queries,
                    base_file="base.fbin",
                    query_file="query.fbin",
                    groundtruth_neighbors_file="groundtruth_neighbors.ibin",
                    distance_metric="euclidean",
                )

                indexes = [
                    IndexConfig(
                        name="elastic_hnsw_integration",
                        algo="elastic_hnsw",
                        build_param={
                            "m": 16,
                            "ef_construction": 64,
                            "use_gpu": False,
                        },
                        search_params=[{"num_candidates": 50}],
                        file="",
                    )
                ]

                # Build
                build_result = backend.build(
                    dataset=dataset,
                    indexes=indexes,
                    force=True,
                    dry_run=False,
                )
                assert build_result.success, build_result.error_message
                assert build_result.index_size_bytes > 0

                # Search (uses data_prefix from config set above)
                search_result = backend.search(
                    dataset=dataset,
                    indexes=indexes,
                    k=k,
                    dry_run=False,
                )
                assert search_result.success, search_result.error_message
                assert search_result.neighbors.shape == (n_queries, k)
                assert search_result.search_time_ms >= 0
        finally:
            backend.cleanup()
