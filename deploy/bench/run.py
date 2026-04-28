#!/usr/bin/env python3
"""
OpenSearch GPU Remote Index Build Benchmark
===========================================
Steps:
  1. Register S3 snapshot repository with OpenSearch
  2. Configure cluster settings for GPU remote index build
  3. Build kNN index via cuvs-bench:
       a. Bulk-ingest dataset vectors
       b. Trigger force-merge to kick off the GPU build
       c. Poll S3 for .faiss files confirming GPU build completion
  4. Run cuvs-bench search benchmarks and print results
  5. Write gbench-compatible JSON result files so cuvs_bench.run --data-export
     and cuvs_bench.plot can be used for CSV export and plotting
"""

import json
import os
import sys

import requests
from cuvs_bench.orchestrator import BenchmarkOrchestrator
from cuvs_bench.backends.base import BuildResult, SearchResult


OPENSEARCH_URL  = os.environ.get("OPENSEARCH_URL",  "http://opensearch:9200")
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "opensearch")
OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", "9200"))
BUILDER_URL     = os.environ.get("BUILDER_URL",     "http://remote-index-builder:1025")

REMOTE_INDEX_BUILD = os.environ.get("REMOTE_INDEX_BUILD", "false").lower() == "true"

S3_BUCKET  = os.environ.get("S3_BUCKET", "")
S3_REGION  = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

DATASET      = os.environ.get("DATASET",       "sift-128-euclidean")
DATASET_PATH = os.environ.get("DATASET_PATH",  "/data/datasets")
BENCH_GROUPS = os.environ.get("BENCH_GROUPS",  "test")
K            = int(os.environ.get("K", "10"))

REPO_NAME = "vector-repo"

session = requests.Session()
session.headers.update({"Content-Type": "application/json"})


# ── helpers ───────────────────────────────────────────────────────────────────

def banner(msg: str) -> None:
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}")


# ── OpenSearch setup ──────────────────────────────────────────────────────────

def register_repository() -> None:
    banner(f"Registering S3 repository '{REPO_NAME}'")
    r = session.put(
        f"{OPENSEARCH_URL}/_snapshot/{REPO_NAME}",
        json={
            "type": "s3",
            "settings": {
                "bucket":    S3_BUCKET,
                "base_path": "knn-indexes",
                "region":    S3_REGION,
            },
        },
    )
    r.raise_for_status()
    print(f"  {r.json()}")


def configure_cluster() -> None:
    if REMOTE_INDEX_BUILD:
        banner("Enabling GPU remote index build (cluster settings)")
        settings = {
            "knn.remote_index_build.enabled":          True,
            "knn.remote_index_build.repository":       REPO_NAME,
            "knn.remote_index_build.service.endpoint": BUILDER_URL,
        }
    else:
        banner("Disabling GPU remote index build (cluster settings)")
        settings = {
            "knn.remote_index_build.enabled": False,
        }
    r = session.put(
        f"{OPENSEARCH_URL}/_cluster/settings",
        json={"persistent": settings},
    )
    r.raise_for_status()
    print(f"  {r.json()}")


# ── result files ─────────────────────────────────────────────────────────────

def write_result_files(
    build_results: list,
    search_results: list,
    dataset: str,
    dataset_path: str,
    algo: str,
    groups: str,
    k: int,
    batch_size: int = 10000,
) -> None:
    """Write gbench-compatible JSON result files.

    Creates files under <dataset_path>/<dataset>/result/{build,search}/ in the
    same format the C++ backend produces, so ``cuvs_bench.run --data-export``
    and ``cuvs_bench.plot`` work without modification.
    """
    build_dir = os.path.join(dataset_path, dataset, "result", "build")
    search_dir = os.path.join(dataset_path, dataset, "result", "search")
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    # Build JSON – one record per successfully built index.
    # data_export.py assumes the build CSV has columns:
    #   [algo_name, index_name, time, threads, cpu_time, ...]
    # so "threads" and "cpu_time" must be present in the JSON (they are not in
    # skip_build_cols and therefore get included as columns 3 and 4).
    build_benchmarks = [
        {
            "name": r.index_path,
            "real_time": r.build_time_seconds,
            "time_unit": "s",
            "threads": 1,
            "cpu_time": r.build_time_seconds,
        }
        for r in build_results
        if isinstance(r, BuildResult) and r.success and r.index_path
    ]

    # Search JSON – zip build + search results to recover the index name, then
    # expand per-search-param entries so each (index, ef_search) is one record.
    build_list  = [r for r in build_results  if isinstance(r, BuildResult)]
    search_list = [r for r in search_results if isinstance(r, SearchResult)]
    search_benchmarks = []
    for build_r, search_r in zip(build_list, search_list):
        if not search_r.success or not build_r.index_path:
            continue
        for entry in (search_r.metadata or {}).get("per_search_param_results", []):
            search_benchmarks.append({
                "name": build_r.index_path,
                "real_time": entry["search_time_ms"],
                "time_unit": "ms",
                "Recall": entry["recall"],
                "items_per_second": entry["queries_per_second"],
                # Latency field expected by data_export in seconds
                "Latency": entry["search_time_ms"] / 1000.0,
            })

    build_file  = os.path.join(build_dir,  f"{algo},{groups}.json")
    search_file = os.path.join(search_dir, f"{algo},{groups},k{k},bs{batch_size}.json")

    with open(build_file, "w") as fh:
        json.dump({"benchmarks": build_benchmarks}, fh, indent=2)
    with open(search_file, "w") as fh:
        json.dump({"benchmarks": search_benchmarks}, fh, indent=2)

    print(f"\n  Result files written:")
    print(f"    {build_file}")
    print(f"    {search_file}")


# ── results ───────────────────────────────────────────────────────────────────

def _print_result_row(params: dict, recall: float, qps: float, latency_ms: float) -> None:
    params_str = ", ".join(f"{k}={v}" for k, v in params.items())
    print(f"  {params_str:<40}  {recall:<12.4f}  {qps:>8.1f}  {latency_ms:>12.2f}")


def print_results(results: list) -> None:
    banner("Benchmark Results")
    search_results = [r for r in results if isinstance(r, SearchResult)]
    if not search_results:
        print("  No search results returned.")
        return

    header = f"  {'params':<40}  {'recall@'+str(K):<12}  {'QPS':>8}  {'latency (ms)':>12}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for r in search_results:
        per_param = (r.metadata or {}).get("per_search_param_results")
        if per_param:
            for entry in per_param:
                _print_result_row(entry["search_params"], entry["recall"], entry["queries_per_second"], entry["search_time_ms"])
        else:
            _print_result_row(r.search_params[0] if r.search_params else {}, r.recall, r.queries_per_second, r.search_time_ms)


# ── entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    if REMOTE_INDEX_BUILD and not S3_BUCKET:
        print("ERROR: S3_BUCKET must be set when REMOTE_INDEX_BUILD=true")
        sys.exit(1)

    print("\n" + "═" * 60)
    print("  OpenSearch kNN Benchmark")
    print("═" * 60)
    print(f"  OpenSearch         : {OPENSEARCH_URL}")
    print(f"  Remote index build : {REMOTE_INDEX_BUILD}")
    if REMOTE_INDEX_BUILD:
        print(f"  GPU builder        : {BUILDER_URL}")
        print(f"  S3 bucket          : s3://{S3_BUCKET}/knn-indexes/  (region: {S3_REGION})")
    print(f"  Dataset            : {DATASET}  (path: {DATASET_PATH})")
    print(f"  Groups             : {BENCH_GROUPS}  k={K}")

    if REMOTE_INDEX_BUILD:
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
        remote_index_build=REMOTE_INDEX_BUILD,
    )
    # ── Build phase ───────────────────────────────────────────────────────────
    mode = "GPU remote build" if REMOTE_INDEX_BUILD else "CPU"
    banner(f"Building index ({mode} via cuvs-bench)")
    build_results = orchestrator.run_benchmark(
        build=True,
        search=False,
        force=True,
        bulk_batch_size=10_000,
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

    write_result_files(
        build_results=build_results,
        search_results=search_results,
        dataset=DATASET,
        dataset_path=DATASET_PATH,
        algo=bench_kwargs["algorithms"],
        groups=BENCH_GROUPS,
        k=K,
    )

    print("\n" + "═" * 60)
    print("  Benchmark complete!")
    print("═" * 60)
    print(f"\n  OpenSearch         : {OPENSEARCH_URL}")
    if REMOTE_INDEX_BUILD:
        print(f"  GPU builder        : {BUILDER_URL}")
    print()


if __name__ == "__main__":
    main()
