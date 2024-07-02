# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import numpy as np
import pytest
from pylibraft.common import device_ndarray
from scipy.spatial.distance import cdist

from cuvs.neighbors import brute_force, filters


@pytest.mark.parametrize("n_index_rows", [32, 100])
@pytest.mark.parametrize("n_query_rows", [32, 100])
@pytest.mark.parametrize("n_cols", [40, 100])
@pytest.mark.parametrize("k", [1, 5, 32])
@pytest.mark.parametrize(
    "metric",
    [
        "euclidean",
        "cityblock",
        "chebyshev",
        "canberra",
        "correlation",
        "russellrao",
        "cosine",
        "sqeuclidean",
        # "inner_product",
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("dtype", [np.float32])
def test_brute_force_knn(
    n_index_rows, n_query_rows, n_cols, k, inplace, metric, dtype
):
    index = np.random.random_sample((n_index_rows, n_cols)).astype(dtype)
    queries = np.random.random_sample((n_query_rows, n_cols)).astype(dtype)

    # RussellRao expects boolean arrays
    if metric == "russellrao":
        index[index < 0.5] = 0.0
        index[index >= 0.5] = 1.0
        queries[queries < 0.5] = 0.0
        queries[queries >= 0.5] = 1.0

    indices = np.zeros((n_query_rows, k), dtype="int64")
    distances = np.zeros((n_query_rows, k), dtype=dtype)

    index_device = device_ndarray(index)
    queries_device = device_ndarray(queries)
    indices_device = device_ndarray(indices)
    distances_device = device_ndarray(distances)

    prefilter = filters.no_filter()

    brute_force_index = brute_force.build(index_device, metric)
    ret_distances, ret_indices = brute_force.search(
        brute_force_index,
        queries_device,
        k,
        neighbors=indices_device,
        distances=distances_device,
        prefilter=prefilter,
    )

    pw_dists = cdist(queries, index, metric=metric)

    distances_device = ret_distances if not inplace else distances_device

    actual_distances = distances_device.copy_to_host()

    actual_distances[actual_distances <= 1e-5] = 0.0
    argsort = np.argsort(pw_dists, axis=1)

    for i in range(pw_dists.shape[0]):
        expected_indices = argsort[i]
        gpu_dists = actual_distances[i]

        cpu_ordered = pw_dists[i, expected_indices]
        np.testing.assert_allclose(
            cpu_ordered[:k], gpu_dists, atol=1e-3, rtol=1e-3
        )


def create_sparse_array(shape, sparsity):
    bits_per_uint32 = 32

    num_bits = np.prod(shape) * bits_per_uint32
    num_ones = int(num_bits * sparsity)

    array = np.zeros(shape, dtype=np.uint32)
    indices = np.random.choice(num_bits, num_ones, replace=False)

    for index in indices:
        i = index // bits_per_uint32
        bit_position = index % bits_per_uint32
        array.flat[i] |= 1 << bit_position

    return array


@pytest.mark.parametrize("n_index_rows", [32, 100])
@pytest.mark.parametrize("n_query_rows", [32, 100])
@pytest.mark.parametrize("n_cols", [40, 100])
@pytest.mark.parametrize("k", [1, 5, 32])
@pytest.mark.parametrize("sparsity", [0.01, 0.2, 0.4])
@pytest.mark.parametrize(
    "metric",
    [
        "euclidean",
        "cosine",
        "sqeuclidean",
    ],
)
@pytest.mark.parametrize("inplace", [True])
@pytest.mark.parametrize("dtype", [np.float32])
def test_prefiltered_brute_force_knn(
    n_index_rows, n_query_rows, n_cols, k, sparsity, inplace, metric, dtype
):
    index = np.random.random_sample((n_index_rows, n_cols)).astype(dtype)
    queries = np.random.random_sample((n_query_rows, n_cols)).astype(dtype)
    bitmap = create_sparse_array(
        (np.ceil(n_query_rows * n_index_rows / 32).astype(int)), sparsity
    )

    is_min = metric != "inner_product"

    indices = np.zeros((n_query_rows, k), dtype="int64")
    distances = np.zeros((n_query_rows, k), dtype=dtype)

    index_device = device_ndarray(index)
    queries_device = device_ndarray(queries)
    indices_device = device_ndarray(indices)
    distances_device = device_ndarray(distances)
    bitmap_device = device_ndarray(bitmap)
    prefilter = filters.from_bitmap(bitmap_device)
    bitmap_device = None
    brute_force_index = brute_force.build(index_device, metric)
    ret_distances, ret_indices = brute_force.search(
        brute_force_index,
        queries_device,
        k,
        neighbors=indices_device,
        distances=distances_device,
        prefilter=prefilter,
    )

    pw_dists = cdist(queries, index, metric=metric)

    # convert bitmap to bool array.
    bitmap_as_uint8 = bitmap.view(np.uint8)
    bool_filter = np.unpackbits(bitmap_as_uint8)
    bool_filter = bool_filter.reshape(-1, 4, 8)
    bool_filter = np.flip(bool_filter, axis=2)
    bool_filter = bool_filter.reshape(-1)[: (n_query_rows * n_index_rows)]
    bool_filter = bool_filter.reshape(-1, n_index_rows)
    bool_filter = np.logical_not(bool_filter)

    pw_dists[bool_filter] = np.inf if is_min else -np.inf

    distances_device = ret_distances if not inplace else distances_device

    actual_distances = distances_device.copy_to_host()

    actual_distances[actual_distances <= 1e-5] = 0.0
    argsort = np.argsort(pw_dists, axis=1)

    for i in range(pw_dists.shape[0]):
        expected_indices = argsort[i]
        gpu_dists = actual_distances[i]

        cpu_ordered = pw_dists[i, expected_indices]
        np.testing.assert_allclose(
            cpu_ordered[:k], gpu_dists, atol=1e-3, rtol=1e-3
        )
