#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OpenSearch GPU Remote Index Build Benchmark
===========================================
Steps:
  1. Register MinIO S3 snapshot repository with OpenSearch
  2. Configure cluster settings for GPU remote index build
  3. Build kNN index via cuvs-bench:
       a. Bulk-ingest dataset vectors
       b. Trigger force-merge to kick off the GPU build
       c. Poll MinIO for .faiss files confirming GPU build completion
  4. Run cuvs-bench search benchmarks and print results
"""

import os
import sys

import requests
from cuvs_bench.orchestrator import BenchmarkOrchestrator
from cuvs_bench.backends.base import BuildResult, SearchResult


OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://opensearch:9200")
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "opensearch")
OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", "9200"))
BUILDER_URL = os.environ.get("BUILDER_URL", "http://remote-index-builder:1025")
MINIO_URL = os.environ.get("MINIO_URL", "http://minio:9000")

DATASET = os.environ.get("DATASET", "sift-128-euclidean")
DATASET_PATH = os.environ.get("DATASET_PATH", "/data/datasets")
BENCH_GROUPS = os.environ.get("BENCH_GROUPS", "test")
K = int(os.environ.get("K", "10"))

BUCKET = "opensearch-vectors"
REPO_NAME = "vector-repo"

session = requests.Session()
session.headers.update({"Content-Type": "application/json"})


# ── helpers ───────────────────────────────────────────────────────────────────


def banner(msg: str) -> None:
    print(f"\n{'─' * 60}\n  {msg}\n{'─' * 60}")


# ── OpenSearch setup ──────────────────────────────────────────────────────────


def register_repository() -> None:
    banner(f"Registering S3 repository '{REPO_NAME}' (backed by MinIO)")
    r = session.put(
        f"{OPENSEARCH_URL}/_snapshot/{REPO_NAME}",
        json={
            "type": "s3",
            "settings": {
                # endpoint / protocol / path_style_access are client-level settings
                # configured in opensearch/opensearch.yml, not per-repository settings.
                "bucket": BUCKET,
                "base_path": "knn-indexes",
            },
        },
    )
    r.raise_for_status()
    print(f"  {r.json()}")


def configure_cluster() -> None:
    banner("Enabling GPU remote index build (cluster settings)")
    r = session.put(
        f"{OPENSEARCH_URL}/_cluster/settings",
        json={
            "persistent": {
                "knn.remote_index_build.enabled": True,
                "knn.remote_index_build.repository": REPO_NAME,
                "knn.remote_index_build.service.endpoint": BUILDER_URL,
            }
        },
    )
    r.raise_for_status()
    print(f"  {r.json()}")


# ── results ───────────────────────────────────────────────────────────────────


def _print_result_row(
    params: dict, recall: float, qps: float, latency_ms: float
) -> None:
    params_str = ", ".join(f"{k}={v}" for k, v in params.items())
    print(
        f"  {params_str:<40}  {recall:<12.4f}  {qps:>8.1f}  {latency_ms:>12.2f}"
    )


def print_results(results: list) -> None:
    banner("Benchmark Results")
    search_results = [r for r in results if isinstance(r, SearchResult)]
    if not search_results:
        print("  No search results returned.")
        return

    header = f"  {'params':<40}  {'recall@' + str(K):<12}  {'QPS':>8}  {'latency (ms)':>12}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for r in search_results:
        per_param = (r.metadata or {}).get("per_search_param_results")
        if per_param:
            for entry in per_param:
                _print_result_row(
                    entry["search_params"],
                    entry["recall"],
                    entry["queries_per_second"],
                    entry["search_time_ms"],
                )
        else:
            _print_result_row(
                r.search_params[0] if r.search_params else {},
                r.recall,
                r.queries_per_second,
                r.search_time_ms,
            )


# ── entrypoint ────────────────────────────────────────────────────────────────


def main() -> None:
    print("\n" + "═" * 60)
    print("  OpenSearch GPU Remote Index Build Benchmark")
    print("═" * 60)
    print(f"  OpenSearch : {OPENSEARCH_URL}")
    print(f"  GPU builder: {BUILDER_URL}")
    print(f"  Dataset    : {DATASET}  (path: {DATASET_PATH})")
    print(f"  Groups     : {BENCH_GROUPS}  k={K}")

    register_repository()
    configure_cluster()

    orchestrator = BenchmarkOrchestrator(backend_type="opensearch")

    # Shared kwargs for both build and search phases
    bench_kwargs = dict(
        dataset=DATASET,
        dataset_path=DATASET_PATH,
        algorithms="opensearch_faiss_hnsw",
        groups=BENCH_GROUPS,
        host=OPENSEARCH_HOST,
        port=OPENSEARCH_PORT,
        use_ssl=False,
        verify_certs=False,
        remote_index_build=True,
        # S3/MinIO config for GPU build verification (used by the backend)
        remote_build_s3_endpoint=MINIO_URL,
        remote_build_s3_bucket=BUCKET,
        remote_build_s3_prefix="knn-indexes/",
        remote_build_s3_access_key="minioadmin",
        remote_build_s3_secret_key="minioadmin",
    )

    # ── Build phase ───────────────────────────────────────────────────────────
    # The backend handles the full GPU build flow: ingest vectors → force merge
    # → poll MinIO for .faiss files confirming GPU build completion.
    banner("Building index (GPU remote build via cuvs-bench)")
    build_results = orchestrator.run_benchmark(
        build=True,
        search=False,
        force=True,
        bulk_batch_size=500,
        **bench_kwargs,
    )

    index_names = [
        r.index_path
        for r in build_results
        if isinstance(r, BuildResult) and r.success and r.index_path
    ]
    if not index_names:
        print("  ERROR: no indexes were successfully built")
        sys.exit(1)
    build_times = {
        r.index_path: r.build_time_seconds
        for r in build_results
        if isinstance(r, BuildResult) and r.success and r.index_path
    }
    for name, t in build_times.items():
        print(f"  {name}: built in {t:.1f}s")

    # ── Search phase ──────────────────────────────────────────────────────────
    banner("Running search benchmarks (via cuvs-bench)")
    search_results = orchestrator.run_benchmark(
        build=False,
        search=True,
        count=K,
        **bench_kwargs,
    )

    print_results(search_results)

    print("\n" + "═" * 60)
    print("  Benchmark complete!")
    print("═" * 60)
    print(f"\n  OpenSearch : {OPENSEARCH_URL}")
    print("  MinIO console : http://localhost:9001  (minioadmin / minioadmin)")
    print()


if __name__ == "__main__":
    main()
