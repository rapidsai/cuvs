#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""
Elasticsearch GPU backend plugin for cuvs-bench.

Install with: pip install cuvs-bench[elastic]

Usage:
    # Build and search as separate operations (matches backend interface)
    from cuvs_bench_elastic import run_build, run_search
    build_results = run_build(dataset="test-data", dataset_path="./datasets", host="localhost", port=9200)
    search_results = run_search(dataset="test-data", dataset_path="./datasets", host="localhost", port=9200)

    # Or run both
    from cuvs_bench_elastic import run_benchmark
    results = run_benchmark(dataset="test-data", dataset_path="./datasets", host="localhost", port=9200)

    # Or use the orchestrator directly
    from cuvs_bench_elastic import ELASTIC, register
    from cuvs_bench.orchestrator import BenchmarkOrchestrator
    register()
    orch = BenchmarkOrchestrator(backend_type=ELASTIC)
    results = orch.run_benchmark(dataset="test-data", dataset_path="./datasets", host="localhost", port=9200)
"""

from .backend import register

# Backend type for use with BenchmarkOrchestrator(backend_type=ELASTIC)
ELASTIC = "elastic"


def _common_kwargs(
    dataset: str,
    dataset_path: str,
    host: str,
    port: int,
    algorithms: str,
    force: bool,
    username: str,
    password: str,
    **kwargs,
):
    """Build kwargs shared by run_build, run_search, run_benchmark."""
    d = dict(
        dataset=dataset,
        dataset_path=dataset_path,
        host=host,
        port=port,
        algorithms=algorithms,
        force=force,
        **kwargs,
    )
    if username:
        d["username"] = username
    if password:
        d["password"] = password
    return d


def run_build(
    dataset: str = "test-data",
    dataset_path: str = "./datasets",
    host: str = "localhost",
    port: int = 9200,
    algorithms: str = "test",
    force: bool = False,
    username: str = None,
    password: str = None,
    **kwargs,
):
    """
    Build an Elasticsearch vector index. Matches the backend build() interface.

    Returns
    -------
    list
        BuildResult objects
    """
    register()
    from cuvs_bench.orchestrator import BenchmarkOrchestrator

    orch = BenchmarkOrchestrator(backend_type=ELASTIC)
    return orch.run_benchmark(
        build=True,
        search=False,
        **_common_kwargs(
            dataset, dataset_path, host, port, algorithms, force, username, password, **kwargs
        ),
    )


def run_search(
    dataset: str = "test-data",
    dataset_path: str = "./datasets",
    host: str = "localhost",
    port: int = 9200,
    algorithms: str = "test",
    username: str = None,
    password: str = None,
    **kwargs,
):
    """
    Run kNN search against an existing Elasticsearch index. Matches the backend search() interface.

    Returns
    -------
    list
        SearchResult objects
    """
    register()
    from cuvs_bench.orchestrator import BenchmarkOrchestrator

    orch = BenchmarkOrchestrator(backend_type=ELASTIC)
    return orch.run_benchmark(
        build=False,
        search=True,
        **_common_kwargs(
            dataset, dataset_path, host, port, algorithms, False, username, password, **kwargs
        ),
    )


def run_benchmark(
    dataset: str = "test-data",
    dataset_path: str = "./datasets",
    host: str = "localhost",
    port: int = 9200,
    algorithms: str = "test",
    build: bool = True,
    search: bool = True,
    force: bool = False,
    username: str = None,
    password: str = None,
    **kwargs,
):
    """
    Run build and/or search. Convenience wrapper; run_build and run_search are the primary API.
    """
    register()
    from cuvs_bench.orchestrator import BenchmarkOrchestrator

    orch = BenchmarkOrchestrator(backend_type=ELASTIC)
    return orch.run_benchmark(
        build=build,
        search=search,
        **_common_kwargs(
            dataset, dataset_path, host, port, algorithms, force, username, password, **kwargs
        ),
    )


__all__ = ["register", "ELASTIC", "run_build", "run_search", "run_benchmark"]
