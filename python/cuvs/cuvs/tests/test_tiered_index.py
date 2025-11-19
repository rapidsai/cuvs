# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.neighbors import (
    brute_force,
    cagra,
    filters,
    ivf_flat,
    ivf_pq,
    tiered_index,
)
from cuvs.tests.ann_utils import calc_recall, create_sparse_bitset


@pytest.mark.parametrize("n_dataset_rows", [1024, 10000])
@pytest.mark.parametrize("n_query_rows", [10])
@pytest.mark.parametrize("n_cols", [10])
@pytest.mark.parametrize("k", [8, 16])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize(
    "metric",
    [
        "sqeuclidean",
        "inner_product",
    ],
)
@pytest.mark.parametrize(
    "algo",
    [
        "cagra",
        "ivf_flat",
        "ivf_pq",
    ],
)
@pytest.mark.parametrize("filter_type", ["bitset_filter", "no_filter"])
def test_tiered_index(
    n_dataset_rows, n_query_rows, n_cols, k, dtype, metric, algo, filter_type
):
    dataset = np.random.random_sample((n_dataset_rows, n_cols)).astype(dtype)
    queries = np.random.random_sample((n_query_rows, n_cols)).astype(dtype)

    indices = np.zeros((n_query_rows, k), dtype="int64")
    distances = np.zeros((n_query_rows, k), dtype="float32")

    dataset_device = device_ndarray(dataset)
    queries_device = device_ndarray(queries)
    indices_device = device_ndarray(indices)
    distances_device = device_ndarray(distances)

    # build with half the dataset, then extend with the other half
    dataset_1_device = device_ndarray(dataset[: n_dataset_rows // 2, :])
    dataset_2_device = device_ndarray(dataset[n_dataset_rows // 2 :, :])

    build_params = tiered_index.IndexParams(
        metric=metric, algo=algo, min_ann_rows=1000
    )
    index = tiered_index.build(build_params, dataset_1_device)
    index = tiered_index.extend(index, dataset_2_device)

    if filter_type == "bitset_filter":
        sparsity = 0.5
        bitset = create_sparse_bitset(n_dataset_rows, sparsity)
        bitset_device = device_ndarray(bitset)
        prefilter = filters.from_bitset(bitset_device)

        # compact the index until we fully support filtered search here
        # index = tiered_index.compact(index)
    else:
        prefilter = filters.no_filter()

    if algo == "cagra":
        search_params = cagra.SearchParams()
    elif algo == "ivf_flat":
        search_params = ivf_flat.SearchParams(n_probes=64)
    elif algo == "ivf_pq":
        search_params = ivf_pq.SearchParams(n_probes=64)

    ret_distances, ret_indices = tiered_index.search(
        search_params,
        index,
        queries_device,
        k,
        neighbors=indices_device,
        distances=distances_device,
        filter=prefilter,
    )

    bfknn_index = brute_force.build(dataset_device, metric)
    groundtruth_neighbors, groundtruth_indices = brute_force.search(
        bfknn_index, queries_device, k, prefilter=prefilter
    )

    ret_indices = ret_indices.copy_to_host()
    groundtruth_indices = groundtruth_indices.copy_to_host()
    recall = calc_recall(ret_indices, groundtruth_indices)
    assert recall > 0.7
