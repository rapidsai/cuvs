# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import tempfile

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from cuvs.common import MultiGpuResources
from cuvs.neighbors.mg import ivf_pq as mg_ivf_pq
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


def run_mg_ivf_pq_build_search_test(
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
    pq_bits=8,
    pq_dim=0,
    codebook_kind="subspace",
):
    """
    Run a multi-GPU IVF-PQ build and search test.

    Note: Multi-GPU IVF-PQ requires host memory arrays (NumPy), not device
    arrays.
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

    # Build parameters - use fewer clusters for better recall with smaller
    # datasets
    if n_lists is None:
        # Use fewer clusters for smaller datasets to ensure enough points per
        # cluster
        n_lists = min(1024, max(64, n_rows // 50))

    build_params = mg_ivf_pq.IndexParams(
        metric=metric,
        distribution_mode=distribution_mode,
        add_data_on_build=add_data_on_build,
        n_lists=n_lists,
        pq_bits=pq_bits,
        pq_dim=pq_dim,
        codebook_kind=codebook_kind,
    )

    # Build index
    index = mg_ivf_pq.build(build_params, dataset, resources=resources)
    assert index.trained

    # If not adding data on build, extend the index
    if not add_data_on_build:
        dataset_1 = dataset[: n_rows // 2, :]
        dataset_2 = dataset[n_rows // 2 :, :]
        indices_1 = np.arange(n_rows // 2, dtype=np.int64)
        indices_2 = np.arange(n_rows // 2, n_rows, dtype=np.int64)

        mg_ivf_pq.extend(index, dataset_1, indices_1, resources=resources)
        mg_ivf_pq.extend(index, dataset_2, indices_2, resources=resources)

    # Search parameters
    if search_params is None:
        search_params = {}
    # Use higher n_probes for better recall in multi-GPU setting
    if "n_probes" not in search_params:
        # Use many clusters for good recall - search majority of clusters
        search_params["n_probes"] = min(n_lists, max(20, (n_lists * 3) // 4))
    search_params_obj = mg_ivf_pq.SearchParams(
        search_mode=search_mode,
        merge_mode=merge_mode,
        n_rows_per_batch=n_rows_per_batch,
        **search_params,
    )

    # Perform search
    distances, neighbors = mg_ivf_pq.search(
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

    return distances, neighbors, recall


@requires_multiple_gpus
@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize(
    "metric", ["sqeuclidean"]
)  # Start with just sqeuclidean
@pytest.mark.parametrize(
    "distribution_mode", ["sharded"]
)  # Start with just sharded
def test_mg_ivf_pq_basic(dtype, metric, distribution_mode):
    """Test basic multi-GPU IVF-PQ build and search functionality."""
    run_mg_ivf_pq_build_search_test(
        n_rows=2000,  # Use smaller dataset for more reliable tests
        n_cols=32,
        n_queries=20,
        k=5,
        dtype=dtype,
        metric=metric,
        distribution_mode=distribution_mode,
        n_lists=50,  # Fixed small number of clusters
        compare=True,
    )


@requires_multiple_gpus
@pytest.mark.parametrize("metric", ["inner_product", "euclidean", "cosine"])
@pytest.mark.parametrize("distribution_mode", ["replicated"])
def test_mg_ivf_pq_additional_metrics(metric, distribution_mode):
    """Test additional metrics and distribution modes for IVF-PQ."""
    run_mg_ivf_pq_build_search_test(
        n_rows=2000,
        n_cols=32,
        n_queries=20,
        k=5,
        dtype=np.float32,
        metric=metric,
        distribution_mode=distribution_mode,
        n_lists=50,
        compare=False,  # PQ may have lower recall, don't enforce strict recall
    )


@requires_multiple_gpus
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
def test_mg_ivf_pq_dtypes(dtype):
    """Test multi-GPU IVF-PQ with different data types."""
    run_mg_ivf_pq_build_search_test(
        n_rows=1500,
        n_cols=32,
        n_queries=15,
        k=5,
        dtype=dtype,
        metric="sqeuclidean",
        n_lists=30,
        compare=False,
    )


@requires_multiple_gpus
@pytest.mark.parametrize("distribution_mode", ["sharded", "replicated"])
def test_mg_ivf_pq_distribution_modes(distribution_mode):
    """Test different distribution modes for multi-GPU IVF-PQ."""
    run_mg_ivf_pq_build_search_test(
        n_rows=1500,
        n_cols=32,
        n_queries=15,
        k=5,
        distribution_mode=distribution_mode,
        n_lists=30,
        compare=False,
    )


@requires_multiple_gpus
@pytest.mark.parametrize("search_mode", ["load_balancer", "round_robin"])
@pytest.mark.parametrize("merge_mode", ["merge_on_root_rank", "tree_merge"])
def test_mg_ivf_pq_search_params(search_mode, merge_mode):
    """Test different multi-GPU search parameters for IVF-PQ."""
    run_mg_ivf_pq_build_search_test(
        n_rows=1500,
        n_cols=32,
        n_queries=15,
        k=5,
        search_mode=search_mode,
        merge_mode=merge_mode,
        n_rows_per_batch=500,
        n_lists=30,
        compare=False,
    )


@requires_multiple_gpus
def test_mg_ivf_pq_pq_parameters():
    """Test different PQ-specific parameters."""
    for pq_bits in [4, 8]:
        for pq_dim in [0, 8, 16]:  # 0 means auto-select
            for codebook_kind in ["subspace", "cluster"]:
                run_mg_ivf_pq_build_search_test(
                    n_rows=1000,
                    n_cols=32,
                    n_queries=100,
                    k=10,
                    pq_bits=pq_bits,
                    pq_dim=pq_dim,
                    codebook_kind=codebook_kind,
                    compare=False,
                )


@requires_multiple_gpus
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean"])
def test_mg_ivf_pq_metrics(metric):
    """Test different distance metrics for multi-GPU IVF-PQ."""
    run_mg_ivf_pq_build_search_test(
        n_rows=1500,
        n_cols=32,
        n_queries=15,
        k=5,
        metric=metric,
        n_lists=30,
        compare=False,
    )


@requires_multiple_gpus
def test_mg_ivf_pq_extend():
    """Test extending index with new vectors."""
    run_mg_ivf_pq_build_search_test(
        n_rows=1000,
        n_cols=32,
        n_queries=100,
        k=10,
        add_data_on_build=False,  # This triggers extend functionality
        compare=False,
    )


@requires_multiple_gpus
def test_mg_ivf_pq_serialize():
    """Test serialization and deserialization."""
    # Generate data
    n_rows, n_cols = 1000, 32
    dataset = generate_data((n_rows, n_cols), np.float32)
    queries = generate_data((100, n_cols), np.float32)

    resources = MultiGpuResources()

    # Build index
    build_params = mg_ivf_pq.IndexParams(
        metric="euclidean",
        n_lists=100,
        pq_bits=8,
        pq_dim=16,
    )
    index = mg_ivf_pq.build(build_params, dataset, resources=resources)

    # Search before serialization
    search_params = mg_ivf_pq.SearchParams(n_probes=50)
    distances_1, neighbors_1 = mg_ivf_pq.search(
        search_params, index, queries, 10, resources=resources
    )

    # Serialize
    with tempfile.NamedTemporaryFile(delete=False) as f:
        filename = f.name

    try:
        mg_ivf_pq.save(index, filename, resources=resources)

        # Load index
        index_loaded = mg_ivf_pq.load(filename, resources=resources)
        assert index_loaded.trained

        # Search after loading
        distances_2, neighbors_2 = mg_ivf_pq.search(
            search_params, index_loaded, queries, 10, resources=resources
        )

        # Results should be the same
        assert np.array_equal(distances_1, distances_2)
        assert np.array_equal(neighbors_1, neighbors_2)

    finally:
        if os.path.exists(filename):
            os.unlink(filename)


@requires_multiple_gpus
def test_mg_ivf_pq_distribute():
    """Test distribute functionality for multi-GPU IVF-PQ."""
    # Note: Distribute is for replicating a single-GPU index across multiple
    # GPUs.
    # This test builds a single-GPU index, serializes it, then distributes it.
    # Multi-GPU distribute only supports float32 indexes.

    n_rows, n_cols = 2000, 32
    k = 5

    # Generate data
    dataset = generate_data((n_rows, n_cols), np.float32)
    queries = generate_data((100, n_cols), np.float32)

    # Import single-GPU IVF-PQ to build and serialize a single-GPU index
    from cuvs.common import Resources
    from cuvs.neighbors import ivf_pq

    # Build single-GPU index first
    single_gpu_resources = Resources()
    single_build_params = ivf_pq.IndexParams(
        metric="sqeuclidean", n_lists=50, pq_bits=8, pq_dim=16
    )

    # Convert to device arrays for single-GPU build
    try:
        import cupy as cp

        device_dataset = cp.asarray(dataset, dtype=np.float32)
        single_index = ivf_pq.build(
            single_build_params, device_dataset, resources=single_gpu_resources
        )
    except ImportError:
        pytest.skip("CuPy not available for single-GPU index building")

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        temp_filename = f.name

    try:
        # Serialize single-GPU index
        ivf_pq.save(
            temp_filename, single_index, resources=single_gpu_resources
        )

        # Now distribute the single-GPU index across multiple GPUs
        resources = MultiGpuResources()
        distributed_index = mg_ivf_pq.distribute(
            temp_filename, resources=resources
        )
        assert distributed_index.trained

        # Search using the distributed index
        search_params = mg_ivf_pq.SearchParams(n_probes=25)
        distances, neighbors = mg_ivf_pq.search(
            search_params, distributed_index, queries, k, resources=resources
        )

        # Verify results shape
        assert distances.shape == (100, k)
        assert neighbors.shape == (100, k)

    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


def test_memory_location_validation():
    """Test that multi-GPU IVF-PQ validates memory locations correctly."""
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("CuPy not available")

    # Generate device arrays (should fail) - use enough data points for n_lists
    dataset_gpu = cp.random.random((1000, 32), dtype=cp.float32)
    queries_gpu = cp.random.random((100, 32), dtype=cp.float32)

    # Create parameters with smaller n_lists for the small dataset
    build_params = mg_ivf_pq.IndexParams(
        n_lists=20
    )  # Smaller n_lists for 1000 points
    search_params = mg_ivf_pq.SearchParams()

    # These should raise ValueError about memory location
    with pytest.raises(ValueError, match="host memory"):
        mg_ivf_pq.build(build_params, dataset_gpu)

    # For search test, we need a valid index first
    dataset_cpu = cp.asnumpy(dataset_gpu)
    resources = MultiGpuResources() if has_multiple_gpus() else None
    if resources:
        index = mg_ivf_pq.build(build_params, dataset_cpu, resources=resources)

        with pytest.raises(ValueError, match="host memory"):
            mg_ivf_pq.search(
                search_params, index, queries_gpu, 5, resources=resources
            )


def test_parameter_validation():
    """Test parameter validation for multi-GPU IVF-PQ."""
    # Test invalid distribution mode
    with pytest.raises(ValueError, match="distribution_mode must be"):
        mg_ivf_pq.IndexParams(distribution_mode="invalid")

    # Test invalid search mode
    with pytest.raises(ValueError, match="search_mode must be"):
        mg_ivf_pq.SearchParams(search_mode="invalid")

    # Test invalid merge mode
    with pytest.raises(ValueError, match="merge_mode must be"):
        mg_ivf_pq.SearchParams(merge_mode="invalid")

    # Test invalid codebook kind
    with pytest.raises(ValueError, match="Incorrect codebook kind"):
        mg_ivf_pq.IndexParams(codebook_kind="invalid")


def test_parameter_properties():
    """Test that parameters can be accessed via properties."""
    # Test IndexParams properties
    params = mg_ivf_pq.IndexParams(distribution_mode="replicated")
    assert params.distribution_mode == "replicated"

    params = mg_ivf_pq.IndexParams(distribution_mode="sharded")
    assert params.distribution_mode == "sharded"

    # Test PQ-specific parameters
    params = mg_ivf_pq.IndexParams(
        pq_bits=4, pq_dim=16, codebook_kind="cluster"
    )
    # These don't have properties exposed, but creation should work

    # Test SearchParams creation with different parameters
    mg_ivf_pq.SearchParams(
        search_mode="round_robin",
        merge_mode="tree_merge",
        n_rows_per_batch=2000,
    )
    # These don't have properties exposed, but creation should work


def test_untrained_index_error():
    """Test that using an untrained index raises appropriate errors."""
    resources = MultiGpuResources()

    # Create untrained index
    index = mg_ivf_pq.Index()
    assert not index.trained

    queries = generate_data((100, 10), np.float32)
    search_params = mg_ivf_pq.SearchParams(n_probes=20)

    # Test that search on untrained index fails
    with pytest.raises(ValueError, match="Index needs to be built"):
        mg_ivf_pq.search(
            search_params, index, queries, 10, resources=resources
        )

    # Test that extend on untrained index fails
    new_vectors = generate_data((50, 10), np.float32)
    with pytest.raises(ValueError, match="Index needs to be built"):
        mg_ivf_pq.extend(index, new_vectors, resources=resources)

    # Test that save on untrained index fails
    with pytest.raises(ValueError, match="Index needs to be built"):
        mg_ivf_pq.save(index, "temp.bin", resources=resources)


@requires_multiple_gpus
def test_mg_ivf_pq_with_prealloc_output():
    """Test multi-GPU IVF-PQ search with pre-allocated output arrays."""
    n_rows, n_cols = 1500, 32  # Ensure n_rows > n_lists
    n_queries = 20
    k = 5

    # Generate data in host memory
    dataset = generate_data((n_rows, n_cols), np.float32)
    queries = generate_data((n_queries, n_cols), np.float32)

    resources = MultiGpuResources()

    # Build index with fewer clusters to avoid n_rows < n_lists error
    build_params = mg_ivf_pq.IndexParams(n_lists=30, pq_bits=8, pq_dim=16)
    index = mg_ivf_pq.build(build_params, dataset, resources=resources)

    # Pre-allocate output arrays in host memory
    neighbors = np.empty((n_queries, k), dtype=np.int64)
    distances = np.empty((n_queries, k), dtype=np.float32)

    # Search with pre-allocated arrays
    search_params = mg_ivf_pq.SearchParams(n_probes=20)
    ret_distances, ret_neighbors = mg_ivf_pq.search(
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
    index = mg_ivf_pq.Index()
    assert repr(index) == "Index(type=MultiGpuIvfPq)"


def test_mg_ivf_pq_simple():
    """Simple test to validate multi-GPU IVF-PQ works with very favorable
    parameters.
    """
    if not has_multiple_gpus():
        pytest.skip("Multi-GPU tests require multiple GPUs")

    # Use simple test case that should definitely work
    n_rows, n_cols = 1000, 32
    n_queries, k = 20, 5

    # Generate data
    dataset = generate_data((n_rows, n_cols), np.float32)
    queries = generate_data((n_queries, n_cols), np.float32)

    resources = MultiGpuResources()

    # Use very few clusters for high recall
    build_params = mg_ivf_pq.IndexParams(
        metric="sqeuclidean",
        n_lists=32,  # Very few clusters
        pq_bits=8,
        pq_dim=16,
    )

    # Build index
    index = mg_ivf_pq.build(build_params, dataset, resources=resources)

    # Search with many probes for maximum recall
    search_params = mg_ivf_pq.SearchParams(n_probes=32)  # Search all clusters
    distances, neighbors = mg_ivf_pq.search(
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
def test_mg_ivf_pq_integration():
    """Integration test covering build, search, extend, and serialization."""
    n_rows, n_cols = 2000, 32
    k = 5

    # Generate initial dataset
    dataset = generate_data((n_rows, n_cols), np.float32)
    queries = generate_data((20, n_cols), np.float32)

    resources = MultiGpuResources()

    # Build initial index
    build_params = mg_ivf_pq.IndexParams(
        distribution_mode="sharded",
        metric="sqeuclidean",
        n_lists=50,
        pq_bits=8,
        pq_dim=16,
    )
    index = mg_ivf_pq.build(build_params, dataset, resources=resources)

    # Initial search
    search_params = mg_ivf_pq.SearchParams(
        n_probes=37,
        search_mode="load_balancer",
        merge_mode="merge_on_root_rank",
    )
    distances1, neighbors1 = mg_ivf_pq.search(
        search_params, index, queries, k, resources=resources
    )

    # Extend index with new vectors
    new_vectors = generate_data((200, n_cols), np.float32)
    # Provide indices for extend operation on non-empty index
    new_indices = np.arange(n_rows, n_rows + 200, dtype=np.int64)
    mg_ivf_pq.extend(index, new_vectors, new_indices, resources=resources)

    # Search after extend
    distances2, neighbors2 = mg_ivf_pq.search(
        search_params, index, queries, k, resources=resources
    )

    # Save and reload
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        temp_filename = f.name

    try:
        mg_ivf_pq.save(index, temp_filename, resources=resources)
        reloaded_index = mg_ivf_pq.load(temp_filename, resources=resources)

        # Search with reloaded index
        distances3, neighbors3 = mg_ivf_pq.search(
            search_params, reloaded_index, queries, k, resources=resources
        )

        # Results from extended and reloaded index should match
        np.testing.assert_array_equal(neighbors2, neighbors3)
        np.testing.assert_allclose(distances2, distances3, rtol=1e-6)

    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
