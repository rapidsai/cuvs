# Copyright (c) 2023-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     h ttp://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from pylibraft.common import device_ndarray
from sklearn.neighbors import NearestNeighbors

from cuvs.neighbors import filters


def generate_data(shape, dtype):
    if dtype == np.byte:
        x = np.random.randint(-127, 128, size=shape, dtype=np.byte)
    elif dtype == np.ubyte:
        x = np.random.randint(0, 255, size=shape, dtype=np.ubyte)
    else:
        x = np.random.random_sample(shape).astype(dtype)

    return x


def calc_recall(ann_idx, true_nn_idx):
    assert ann_idx.shape == true_nn_idx.shape
    n = 0
    for i in range(ann_idx.shape[0]):
        n += np.intersect1d(ann_idx[i, :], true_nn_idx[i, :]).size
    recall = n / ann_idx.size
    return recall


def create_sparse_bitset(n_size, sparsity):
    bits_per_uint32 = 32
    num_bits = n_size
    num_uint32s = (num_bits + bits_per_uint32 - 1) // bits_per_uint32
    num_ones = int(num_bits * sparsity)

    array = np.zeros(num_uint32s, dtype=np.uint32)
    indices = np.random.choice(num_bits, num_ones, replace=False)

    for index in indices:
        i = index // bits_per_uint32
        bit_position = index % bits_per_uint32
        array[i] |= 1 << bit_position

    return array


def run_filtered_search_test(
    search_module,
    sparsity,
    n_rows=10000,
    n_cols=10,
    n_queries=10,
    k=10,
):
    dataset = generate_data((n_rows, n_cols), np.float32)
    queries = generate_data((n_queries, n_cols), np.float32)

    bitset = create_sparse_bitset(n_rows, sparsity)

    dataset_device = device_ndarray(dataset)
    queries_device = device_ndarray(queries)
    bitset_device = device_ndarray(bitset)

    build_params = search_module.IndexParams()
    index = search_module.build(build_params, dataset_device)

    filter_ = filters.from_bitset(bitset_device)

    search_params = search_module.SearchParams()
    ret_distances, ret_indices = search_module.search(
        search_params,
        index,
        queries_device,
        k,
        filter=filter_,
    )

    # Convert bitset to bool array for validation
    bitset_as_uint8 = bitset.view(np.uint8)
    bool_filter = np.unpackbits(bitset_as_uint8)
    bool_filter = bool_filter.reshape(-1, 4, 8)
    bool_filter = np.flip(bool_filter, axis=2)
    bool_filter = bool_filter.reshape(-1)[:n_rows]
    bool_filter = np.logical_not(bool_filter)  # Flip so True means filtered

    # Get filtered dataset for reference calculation
    non_filtered_mask = ~bool_filter
    filtered_dataset = dataset[non_filtered_mask]

    nn_skl = NearestNeighbors(
        n_neighbors=k, algorithm="brute", metric="euclidean"
    )
    nn_skl.fit(filtered_dataset)
    skl_idx = nn_skl.kneighbors(queries, return_distance=False)

    actual_indices = ret_indices.copy_to_host()

    filtered_idx_map = (
        np.cumsum(~bool_filter) - 1
    )  # -1 because cumsum starts at 1

    # Map ANN indices to filtered space
    mapped_actual_indices = np.take(
        filtered_idx_map, actual_indices, mode="clip"
    )

    filtered_indices = np.where(bool_filter)[0]
    for i in range(n_queries):
        assert not np.intersect1d(filtered_indices, actual_indices[i]).size

    recall = calc_recall(mapped_actual_indices, skl_idx)

    assert recall > 0.7
