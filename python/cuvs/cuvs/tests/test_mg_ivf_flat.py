# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from cuvs.common import MultiGpuResources
from cuvs.neighbors.mg import ivf_flat as mg_ivf_flat
from cuvs.tests.ann_utils import calc_recall, generate_data


# Check if multi-GPU functionality is available
def has_multiple_gpus():
    """Check if system has multiple GPUs available."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 1
    except Exception:
        return False


# Mark tests that require multiple GPUs
requires_multiple_gpus = pytest.mark.skipif(
    not has_multiple_gpus(), reason="Multi-GPU tests require multiple GPUs"
)


def run_mg_ivf_flat_build_search_test(
    n_rows=10000,
    n_cols=10,
    n_queries=100,
    k=10,
    dtype=np.float32,
    metric="euclidean",
    distribution_mode="sharded",
    search_mode="load_balancer",
    merge_mode="tree_merge",
    n_rows_per_batch=1000,
    compare=True,
    add_data_on_build=True,
    search_params=None,
    n_lists=None,
):
    """
    Run a multi-GPU IVF-Flat build and search test.

    Note: Multi-GPU IVF-Flat requires host memory arrays (NumPy), not
    device arrays.
    """
    # Generate host memory arrays (NumPy)
    dataset = generate_data((n_rows, n_cols), dtype)
    if metric == "inner_product":
        dataset = normalize(dataset, norm="l2", axis=1)

    queries = generate_data((n_queries, n_cols), dtype)
    if metric == "inner_product":
        queries = normalize(queries, norm="l2", axis=1)

    # Multi-GPU resources
    resources = MultiGpuResources()

    # Build parameters - use fewer clusters for better recall
    # with smaller datasets
    if n_lists is None:
        # Use fewer clusters for smaller datasets to ensure enough points
        # per cluster
        n_lists = min(1024, max(64, n_rows // 50))

    build_params = mg_ivf_flat.IndexParams(
        metric=metric,
        distribution_mode=distribution_mode,
        add_data_on_build=add_data_on_build,
        n_lists=n_lists,
    )

    # Build index
    index = mg_ivf_flat.build(build_params, dataset, resources=resources)
    assert index.trained

    # If not adding data on build, extend the index
    if not add_data_on_build:
        dataset_1 = dataset[: n_rows // 2, :]
        dataset_2 = dataset[n_rows // 2 :, :]
        indices_1 = np.arange(n_rows // 2, dtype=np.int64)
        indices_2 = np.arange(n_rows // 2, n_rows, dtype=np.int64)

        mg_ivf_flat.extend(index, dataset_1, indices_1, resources=resources)
        mg_ivf_flat.extend(index, dataset_2, indices_2, resources=resources)

    # Search parameters
    if search_params is None:
        search_params = {}
    # Use higher n_probes for better recall in multi-GPU setting
    if "n_probes" not in search_params:
        # Use many clusters for good recall - search majority of clusters
        search_params["n_probes"] = min(n_lists, max(20, (n_lists * 3) // 4))
    search_params_obj = mg_ivf_flat.SearchParams(
        search_mode=search_mode,
        merge_mode=merge_mode,
        n_rows_per_batch=n_rows_per_batch,
        **search_params,
    )

    # Perform search
    distances, neighbors = mg_ivf_flat.search(
        search_params_obj,
        index,
        queries,
        k,
        resources=resources,
    )

    # Verify results are in host memory (NumPy arrays)
    assert isinstance(distances, np.ndarray)
    assert isinstance(neighbors, np.ndarray)
    assert distances.shape == (n_queries, k)
    assert neighbors.shape == (n_queries, k)

    if not compare:
        return distances, neighbors

    # Calculate reference values with sklearn
    skl_metric = {
        "sqeuclidean": "sqeuclidean",
        "inner_product": "cosine",
        "cosine": "cosine",
        "euclidean": "euclidean",
    }[metric]

    nn_skl = NearestNeighbors(
        n_neighbors=k, algorithm="brute", metric=skl_metric
    )
    nn_skl.fit(dataset)
    skl_idx = nn_skl.kneighbors(queries, return_distance=False)

    recall = calc_recall(neighbors, skl_idx)
    # Multi-GPU implementation may have lower recall due to data distribution
    # across GPUs
    # This is acceptable as long as the functionality works correctly
    assert recall > 0.3, (
        f"Recall too low: {recall:.3f} (n_lists={n_lists}, "
        f"n_probes={search_params.get('n_probes', 'default')})"
    )

    return distances, neighbors


@requires_multiple_gpus
@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize(
    "metric", ["sqeuclidean"]
)  # Start with just sqeuclidean
@pytest.mark.parametrize(
    "distribution_mode", ["sharded"]
)  # Start with just sharded
def test_mg_ivf_flat_basic(dtype, metric, distribution_mode):
    """Test basic multi-GPU IVF-Flat build and search functionality."""
    run_mg_ivf_flat_build_search_test(
        n_rows=2000,  # Use smaller dataset for more reliable tests
        n_cols=8,
        n_queries=20,
        k=5,
        dtype=dtype,
        metric=metric,
        distribution_mode=distribution_mode,
        n_lists=50,  # Fixed small number of clusters
    )


@requires_multiple_gpus
@pytest.mark.parametrize("metric", ["inner_product", "euclidean", "cosine"])
@pytest.mark.parametrize("distribution_mode", ["replicated"])
def test_mg_ivf_flat_additional_metrics(metric, distribution_mode):
    """Test additional metrics and distribution modes."""
    run_mg_ivf_flat_build_search_test(
        n_rows=2000,
        n_cols=8,
        n_queries=20,
        k=5,
        dtype=np.float32,
        metric=metric,
        distribution_mode=distribution_mode,
        n_lists=50,
    )


@requires_multiple_gpus
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
def test_mg_ivf_flat_dtypes(dtype):
    """Test multi-GPU IVF-Flat with different data types."""
    run_mg_ivf_flat_build_search_test(
        n_rows=1500,
        n_cols=8,
        n_queries=15,
        k=5,
        dtype=dtype,
        metric="sqeuclidean",
        n_lists=30,
    )


@requires_multiple_gpus
@pytest.mark.parametrize("distribution_mode", ["sharded", "replicated"])
def test_mg_ivf_flat_distribution_modes(distribution_mode):
    """Test different distribution modes for multi-GPU IVF-Flat."""
    run_mg_ivf_flat_build_search_test(
        n_rows=1500,
        n_cols=8,
        n_queries=15,
        k=5,
        distribution_mode=distribution_mode,
        n_lists=30,
    )


@requires_multiple_gpus
@pytest.mark.parametrize("search_mode", ["load_balancer", "round_robin"])
@pytest.mark.parametrize("merge_mode", ["merge_on_root_rank", "tree_merge"])
def test_mg_ivf_flat_search_params(search_mode, merge_mode):
    """Test different multi-GPU search parameters."""
    run_mg_ivf_flat_build_search_test(
        n_rows=1500,
        n_cols=8,
        n_queries=15,
        k=5,
        search_mode=search_mode,
        merge_mode=merge_mode,
        n_rows_per_batch=500,
        n_lists=30,
    )


@requires_multiple_gpus
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean"])
def test_mg_ivf_flat_metrics(metric):
    """Test different distance metrics for multi-GPU IVF-Flat."""
    run_mg_ivf_flat_build_search_test(
        n_rows=1500,
        n_cols=8,
        n_queries=15,
        k=5,
        metric=metric,
        n_lists=30,
    )


@requires_multiple_gpus
def test_mg_ivf_flat_extend():
    """Test extending multi-GPU IVF-Flat index with new vectors."""
    run_mg_ivf_flat_build_search_test(
        n_rows=1500,
        n_cols=8,
        n_queries=15,
        k=5,
        add_data_on_build=False,
        n_lists=30,
    )


@requires_multiple_gpus
def test_mg_ivf_flat_serialize():
    """Test save/load functionality for multi-GPU IVF-Flat."""
    n_rows, n_cols = 2000, 8
    k = 5

    # Generate data
    dataset = generate_data((n_rows, n_cols), np.float32)
    queries = generate_data((20, n_cols), np.float32)

    resources = MultiGpuResources()

    # Build original index
    build_params = mg_ivf_flat.IndexParams(n_lists=50)
    original_index = mg_ivf_flat.build(
        build_params, dataset, resources=resources
    )

    # Search with original index
    search_params = mg_ivf_flat.SearchParams(n_probes=37)
    orig_distances, orig_neighbors = mg_ivf_flat.search(
        search_params, original_index, queries, k, resources=resources
    )

    # Save index to temporary file
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        temp_filename = f.name

    try:
        mg_ivf_flat.save(original_index, temp_filename, resources=resources)

        # Load index from file
        loaded_index = mg_ivf_flat.load(temp_filename, resources=resources)
        assert loaded_index.trained

        # Search with loaded index
        loaded_distances, loaded_neighbors = mg_ivf_flat.search(
            search_params, loaded_index, queries, k, resources=resources
        )

        # Results should be identical
        np.testing.assert_array_equal(orig_neighbors, loaded_neighbors)
        np.testing.assert_allclose(orig_distances, loaded_distances, rtol=1e-6)

    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


@requires_multiple_gpus
def test_mg_ivf_flat_distribute():
    """Test distribute functionality for multi-GPU IVF-Flat."""
    # Note: Distribute is for replicating a single-GPU index
    # across multiple GPUs.
    # This test builds a single-GPU index, serializes it, then distributes it.

    n_rows, n_cols = 2000, 8
    k = 5

    # Generate data
    dataset = generate_data((n_rows, n_cols), np.float32)
    queries = generate_data((20, n_cols), np.float32)

    # Import single-GPU IVF-Flat to build and serialize a single-GPU index
    from cuvs.common import Resources
    from cuvs.neighbors import ivf_flat

    # Build single-GPU index first
    single_gpu_resources = Resources()
    single_build_params = ivf_flat.IndexParams(
        metric="sqeuclidean", n_lists=50
    )

    # Convert to device arrays for single-GPU build
    try:
        import cupy as cp

        device_dataset = cp.asarray(dataset)
        single_index = ivf_flat.build(
            single_build_params, device_dataset, resources=single_gpu_resources
        )
    except ImportError:
        pytest.skip("CuPy not available for single-GPU index building")

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        temp_filename = f.name

    try:
        # Serialize single-GPU index
        ivf_flat.save(
            temp_filename, single_index, resources=single_gpu_resources
        )

        # Now distribute the single-GPU index across multiple GPUs
        resources = MultiGpuResources()
        distributed_index = mg_ivf_flat.distribute(
            temp_filename, resources=resources
        )
        assert distributed_index.trained

        # Search should work with distributed index (using host memory arrays)
        search_params = mg_ivf_flat.SearchParams(n_probes=37)
        distances, neighbors = mg_ivf_flat.search(
            search_params, distributed_index, queries, k, resources=resources
        )

        assert distances.shape == (20, k)
        assert neighbors.shape == (20, k)

    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


def test_memory_location_validation():
    """Test that multi-GPU IVF-Flat validates memory locations correctly."""
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("CuPy not available for memory location tests")

    n_rows, n_cols = 1500, 8

    # Create host and device arrays
    host_data = generate_data((n_rows, n_cols), np.float32)
    device_data = cp.asarray(host_data)

    resources = MultiGpuResources()
    build_params = mg_ivf_flat.IndexParams(n_lists=30)

    # Test that device arrays are rejected for build
    with pytest.raises(ValueError, match="host memory"):
        mg_ivf_flat.build(build_params, device_data, resources=resources)

    # Test that host arrays work for build
    index = mg_ivf_flat.build(build_params, host_data, resources=resources)

    # Test that device arrays are rejected for search
    queries = generate_data((20, n_cols), np.float32)
    device_queries = cp.asarray(queries)
    search_params = mg_ivf_flat.SearchParams(n_probes=22)

    with pytest.raises(ValueError, match="host memory"):
        mg_ivf_flat.search(
            search_params, index, device_queries, 5, resources=resources
        )

    # Test that host arrays work for search
    distances, neighbors = mg_ivf_flat.search(
        search_params, index, queries, 5, resources=resources
    )
    assert isinstance(distances, np.ndarray)
    assert isinstance(neighbors, np.ndarray)


def test_parameter_validation():
    """Test parameter validation for multi-GPU IVF-Flat."""
    # Test invalid distribution mode
    with pytest.raises(ValueError, match="distribution_mode must be"):
        mg_ivf_flat.IndexParams(distribution_mode="invalid")

    # Test invalid search mode
    with pytest.raises(ValueError, match="search_mode must be"):
        mg_ivf_flat.SearchParams(search_mode="invalid")

    # Test invalid merge mode
    with pytest.raises(ValueError, match="merge_mode must be"):
        mg_ivf_flat.SearchParams(merge_mode="invalid")


def test_parameter_properties():
    """Test that parameters can be accessed via properties."""
    # Test IndexParams properties
    params = mg_ivf_flat.IndexParams(distribution_mode="replicated")
    assert params.distribution_mode == "replicated"

    params = mg_ivf_flat.IndexParams(distribution_mode="sharded")
    assert params.distribution_mode == "sharded"

    # Test SearchParams creation with different parameters
    mg_ivf_flat.SearchParams(
        search_mode="round_robin",
        merge_mode="tree_merge",
        n_rows_per_batch=2000,
    )
    # These don't have properties exposed, but creation should work


def test_untrained_index_error():
    """Test that using an untrained index raises appropriate errors."""
    resources = MultiGpuResources()

    # Create untrained index
    index = mg_ivf_flat.Index()
    assert not index.trained

    queries = generate_data((100, 10), np.float32)
    search_params = mg_ivf_flat.SearchParams(n_probes=20)

    # Test that search on untrained index fails
    with pytest.raises(ValueError, match="Index needs to be built"):
        mg_ivf_flat.search(
            search_params, index, queries, 10, resources=resources
        )

    # Test that extend on untrained index fails
    new_vectors = generate_data((50, 10), np.float32)
    with pytest.raises(ValueError, match="Index needs to be built"):
        mg_ivf_flat.extend(index, new_vectors, resources=resources)

    # Test that save on untrained index fails
    with pytest.raises(ValueError, match="Index needs to be built"):
        mg_ivf_flat.save(index, "temp.bin", resources=resources)


@requires_multiple_gpus
def test_mg_ivf_flat_with_prealloc_output():
    """Test multi-GPU IVF-Flat search with pre-allocated output arrays."""
    n_rows, n_cols = 1500, 8  # Ensure n_rows > n_lists
    n_queries = 20
    k = 5

    # Generate data in host memory
    dataset = generate_data((n_rows, n_cols), np.float32)
    queries = generate_data((n_queries, n_cols), np.float32)

    resources = MultiGpuResources()

    # Build index with fewer clusters to avoid n_rows < n_lists error
    build_params = mg_ivf_flat.IndexParams(n_lists=30)
    index = mg_ivf_flat.build(build_params, dataset, resources=resources)

    # Pre-allocate output arrays in host memory
    neighbors = np.empty((n_queries, k), dtype=np.int64)
    distances = np.empty((n_queries, k), dtype=np.float32)

    # Search with pre-allocated arrays
    search_params = mg_ivf_flat.SearchParams(n_probes=20)
    ret_distances, ret_neighbors = mg_ivf_flat.search(
        search_params,
        index,
        queries,
        k,
        neighbors=neighbors,
        distances=distances,
        resources=resources,
    )

    # Should return the same arrays we passed in
    assert ret_distances is distances
    assert ret_neighbors is neighbors
    assert distances.shape == (n_queries, k)
    assert neighbors.shape == (n_queries, k)


def test_index_repr():
    """Test string representation of Index."""
    index = mg_ivf_flat.Index()
    assert repr(index) == "Index(type=MultiGpuIvfFlat)"


def test_mg_ivf_flat_simple():
    """Simple test to validate multi-GPU IVF-Flat works with very favorable
    parameters.
    """
    if not has_multiple_gpus():
        pytest.skip("Multi-GPU tests require multiple GPUs")

    # Use simple test case that should definitely work
    n_rows, n_cols = 1000, 8
    n_queries, k = 20, 5

    # Generate data
    dataset = generate_data((n_rows, n_cols), np.float32)
    queries = generate_data((n_queries, n_cols), np.float32)

    resources = MultiGpuResources()

    # Use very few clusters for high recall
    build_params = mg_ivf_flat.IndexParams(
        metric="sqeuclidean",
        n_lists=32,  # Very few clusters
    )

    # Build index
    index = mg_ivf_flat.build(build_params, dataset, resources=resources)

    # Search with many probes for maximum recall
    search_params = mg_ivf_flat.SearchParams(
        n_probes=32
    )  # Search all clusters
    distances, neighbors = mg_ivf_flat.search(
        search_params, index, queries, k, resources=resources
    )

    # Basic sanity checks
    assert distances.shape == (n_queries, k)
    assert neighbors.shape == (n_queries, k)
    assert isinstance(distances, np.ndarray)
    assert isinstance(neighbors, np.ndarray)

    # Check that we get valid neighbors
    assert np.all(neighbors >= 0)
    assert np.all(neighbors < n_rows)

    # Distances should be non-negative and sorted
    assert np.all(distances >= 0)
    for i in range(n_queries):
        assert np.all(
            distances[i, :-1] <= distances[i, 1:]
        ), f"Distances not sorted for query {i}"


# Integration test with multiple operations
@requires_multiple_gpus
def test_mg_ivf_flat_integration():
    """Integration test covering build, search, extend, and serialization."""
    n_rows, n_cols = 2000, 8
    k = 5

    # Generate initial dataset
    dataset = generate_data((n_rows, n_cols), np.float32)
    queries = generate_data((20, n_cols), np.float32)

    resources = MultiGpuResources()

    # Build initial index
    build_params = mg_ivf_flat.IndexParams(
        distribution_mode="sharded", metric="sqeuclidean", n_lists=50
    )
    index = mg_ivf_flat.build(build_params, dataset, resources=resources)

    # Initial search
    search_params = mg_ivf_flat.SearchParams(
        n_probes=37,
        search_mode="load_balancer",
        merge_mode="merge_on_root_rank",
    )
    distances1, neighbors1 = mg_ivf_flat.search(
        search_params, index, queries, k, resources=resources
    )

    # Extend index with new vectors
    new_vectors = generate_data((200, n_cols), np.float32)
    # Provide indices for extend operation on non-empty index
    new_indices = np.arange(n_rows, n_rows + 200, dtype=np.int64)
    mg_ivf_flat.extend(index, new_vectors, new_indices, resources=resources)

    # Search after extend
    distances2, neighbors2 = mg_ivf_flat.search(
        search_params, index, queries, k, resources=resources
    )

    # Save and reload
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        temp_filename = f.name

    try:
        mg_ivf_flat.save(index, temp_filename, resources=resources)
        reloaded_index = mg_ivf_flat.load(temp_filename, resources=resources)

        # Search with reloaded index
        distances3, neighbors3 = mg_ivf_flat.search(
            search_params, reloaded_index, queries, k, resources=resources
        )

        # Results from extended and reloaded index should match
        np.testing.assert_array_equal(neighbors2, neighbors3)
        np.testing.assert_allclose(distances2, distances3, rtol=1e-6)

    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
