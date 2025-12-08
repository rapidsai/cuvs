# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from cuvs.neighbors import cagra, hnsw
from cuvs.tests.ann_utils import calc_recall, generate_data


def run_cagra_ace_build_search_test(
    n_rows=5000,
    n_cols=64,
    n_queries=10,
    k=10,
    dtype=np.float32,
    metric="sqeuclidean",
    intermediate_graph_degree=128,
    graph_degree=64,
    npartitions=2,
    ef_construction=100,
    hierarchy="none",
):
    dataset = generate_data((n_rows, n_cols), dtype)
    queries = generate_data((n_queries, n_cols), dtype)
    if metric == "inner_product":
        dataset = normalize(dataset, norm="l2", axis=1)
        queries = normalize(queries, norm="l2", axis=1)
        if dtype in [np.int8, np.uint8]:
            # Quantize the normalized data to the int8/uint8 range
            dtype_max = np.iinfo(dtype).max
            dataset = (dataset * dtype_max).astype(dtype)
            queries = (queries * dtype_max).astype(dtype)

    # Create a temporary directory for ACE build
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up ACE parameters
        ace_params = cagra.AceParams(
            npartitions=npartitions,
            ef_construction=ef_construction,
            build_dir=temp_dir,
        )

        # Build parameters
        build_params = cagra.IndexParams(
            metric=metric,
            intermediate_graph_degree=intermediate_graph_degree,
            graph_degree=graph_degree,
            build_algo="ace",
            ace_params=ace_params,
        )

        # Build the index with ACE (uses disk-based storage)
        index = cagra.build(build_params, dataset)

        assert index.trained

        # Verify that expected files were created
        assert os.path.exists(os.path.join(temp_dir, "cagra_graph.npy"))
        assert os.path.exists(os.path.join(temp_dir, "reordered_dataset.npy"))
        assert os.path.exists(os.path.join(temp_dir, "dataset_mapping.npy"))

        # Test HNSW conversion from disk-based ACE index
        hnsw_params = hnsw.IndexParams(hierarchy=hierarchy)
        hnsw_index_serialized = hnsw.from_cagra(hnsw_params, index)
        assert hnsw_index_serialized is not None
        assert os.path.exists(os.path.join(temp_dir, "hnsw_index.bin"))

        # Deserialize the HNSW index from disk for search
        hnsw_index = hnsw.load(
            hnsw_params,
            os.path.join(temp_dir, "hnsw_index.bin"),
            n_cols,
            dtype,
        )

        search_params = hnsw.SearchParams(ef=200, num_threads=1)
        out_dist, out_idx = hnsw.search(search_params, hnsw_index, queries, k)

        # Calculate reference values with sklearn
        skl_metric = {
            "sqeuclidean": "sqeuclidean",
            "inner_product": "cosine",
            "euclidean": "euclidean",
        }[metric]
        nn_skl = NearestNeighbors(
            n_neighbors=k, algorithm="brute", metric=skl_metric
        )
        nn_skl.fit(dataset)
        skl_dist, skl_idx = nn_skl.kneighbors(queries, return_distance=True)

        recall = calc_recall(out_idx, skl_idx)
        assert recall > 0.7


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
@pytest.mark.parametrize("metric", ["sqeuclidean", "inner_product"])
@pytest.mark.parametrize("npartitions", [2, 4])
@pytest.mark.parametrize("ef_construction", [100, 200])
@pytest.mark.parametrize("hierarchy", ["none", "gpu"])
def test_cagra_ace_dtypes_and_metrics(
    dim, dtype, metric, npartitions, ef_construction, hierarchy
):
    """Test ACE with different data types and metrics."""
    run_cagra_ace_build_search_test(
        n_cols=dim,
        dtype=dtype,
        metric=metric,
        npartitions=npartitions,
        ef_construction=ef_construction,
        hierarchy=hierarchy,
    )


def test_cagra_ace_with_memory_limits():
    """Test ACE build with custom memory limits.

    ACE always uses disk-based storage, but memory limits help control
    the number of partitions to fit within available memory.
    """
    n_rows = 5000
    n_cols = 64
    dtype = np.float32
    metric = "sqeuclidean"

    dataset = generate_data((n_rows, n_cols), dtype)

    # Create a temporary directory for ACE build
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set ACE parameters with memory limits
        ace_params = cagra.AceParams(
            npartitions=2,
            ef_construction=100,
            build_dir=temp_dir,
            max_host_memory_gb=0.001,  # Tiny limit to force more partitions
            max_gpu_memory_gb=0.001,  # Tiny limit to force more partitions
        )

        # Build parameters
        build_params = cagra.IndexParams(
            metric=metric,
            intermediate_graph_degree=128,
            graph_degree=64,
            build_algo="ace",
            ace_params=ace_params,
        )

        # Build the index with ACE (always uses disk mode)
        index = cagra.build(build_params, dataset)

        assert index.trained

        # ACE always stores files to disk
        graph_file = os.path.join(temp_dir, "cagra_graph.npy")
        reordered_file = os.path.join(temp_dir, "reordered_dataset.npy")

        assert os.path.exists(graph_file), (
            "Graph file should exist for ACE build"
        )
        assert os.path.exists(reordered_file), (
            "Reordered dataset file should exist for ACE build"
        )
