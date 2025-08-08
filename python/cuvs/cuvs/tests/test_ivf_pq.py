# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import cupy as cp
import numpy as np
import pytest
from pylibraft.common import device_ndarray
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from cuvs.neighbors import ivf_pq
from cuvs.tests.ann_utils import calc_recall, generate_data


def run_ivf_pq_build_search_test(
    n_rows=10000,
    n_cols=10,
    n_queries=100,
    k=10,
    dtype=np.float32,
    n_lists=100,
    metric="euclidean",
    pq_bits=8,
    pq_dim=0,
    codebook_kind="subspace",
    add_data_on_build=True,
    n_probes=100,
    lut_dtype=np.float32,
    internal_distance_dtype=np.float32,
    force_random_rotation=False,
    kmeans_trainset_fraction=1,
    kmeans_n_iters=20,
    compare=True,
    inplace=True,
    array_type="device",
):
    dataset = generate_data((n_rows, n_cols), dtype)
    if metric == "inner_product":
        dataset = normalize(dataset, norm="l2", axis=1)
    dataset_device = device_ndarray(dataset)

    build_params = ivf_pq.IndexParams(
        n_lists=n_lists,
        metric=metric,
        kmeans_n_iters=kmeans_n_iters,
        kmeans_trainset_fraction=kmeans_trainset_fraction,
        pq_bits=pq_bits,
        pq_dim=pq_dim,
        codebook_kind=codebook_kind,
        force_random_rotation=force_random_rotation,
        add_data_on_build=add_data_on_build,
    )

    if array_type == "device":
        index = ivf_pq.build(build_params, dataset_device)
    else:
        index = ivf_pq.build(build_params, dataset)

    if not add_data_on_build:
        dataset_1 = dataset[: n_rows // 2, :]
        dataset_2 = dataset[n_rows // 2 :, :]
        indices_1 = np.arange(n_rows // 2, dtype=np.int64)
        indices_2 = np.arange(n_rows // 2, n_rows, dtype=np.int64)

        if array_type == "device":
            dataset_1_device = device_ndarray(dataset_1)
            dataset_2_device = device_ndarray(dataset_2)
            indices_1_device = device_ndarray(indices_1)
            indices_2_device = device_ndarray(indices_2)
            index = ivf_pq.extend(index, dataset_1_device, indices_1_device)
            index = ivf_pq.extend(index, dataset_2_device, indices_2_device)
        else:
            index = ivf_pq.extend(index, dataset_1, indices_1)
            index = ivf_pq.extend(index, dataset_2, indices_2)

    queries = generate_data((n_queries, n_cols), dtype)
    out_idx = np.zeros((n_queries, k), dtype=np.int64)
    out_dist = np.zeros((n_queries, k), dtype=np.float32)

    queries_device = device_ndarray(queries)
    out_idx_device = device_ndarray(out_idx) if inplace else None
    out_dist_device = device_ndarray(out_dist) if inplace else None

    search_params = ivf_pq.SearchParams(
        n_probes=n_probes,
        lut_dtype=lut_dtype,
        internal_distance_dtype=internal_distance_dtype,
    )

    ret_output = ivf_pq.search(
        search_params,
        index,
        queries_device,
        k,
        neighbors=out_idx_device,
        distances=out_dist_device,
    )

    if not inplace:
        out_dist_device, out_idx_device = ret_output

    centers = index.centers
    assert centers.shape[0] == n_lists

    if not compare:
        return

    out_idx = out_idx_device.copy_to_host()
    out_dist = out_dist_device.copy_to_host()

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
    skl_idx = nn_skl.kneighbors(queries, return_distance=False)

    recall = calc_recall(out_idx, skl_idx)
    assert recall > 0.7


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize(
    "metric", ["sqeuclidean", "inner_product", "euclidean"]
)
def test_ivf_pq(inplace, dtype, metric):
    run_ivf_pq_build_search_test(
        dtype=dtype,
        inplace=inplace,
        metric=metric,
    )


@pytest.mark.parametrize(
    "params",
    [
        {
            "k": 10,
            "n_probes": 100,
            "lut": np.float16,
            "idd": np.float32,
        },
        {
            "k": 10,
            "n_probes": 99,
            "lut": np.uint8,
            "idd": np.float32,
        },
        {
            "k": 10,
            "n_probes": 100,
            "lut": np.float16,
            "idd": np.float16,
        },
        {
            "k": 129,
            "n_probes": 100,
            "lut": np.float32,
            "idd": np.float32,
        },
    ],
)
def test_ivf_pq_search_params(params):
    run_ivf_pq_build_search_test(
        n_rows=10000,
        n_cols=16,
        n_queries=1000,
        k=params["k"],
        n_lists=100,
        n_probes=params["n_probes"],
        metric="sqeuclidean",
        dtype=np.float32,
        lut_dtype=params["lut"],
        internal_distance_dtype=params["idd"],
    )


@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize("array_type", ["host", "device"])
def test_extend(dtype, array_type):
    run_ivf_pq_build_search_test(
        n_rows=10000,
        n_cols=10,
        n_queries=100,
        k=10,
        n_lists=100,
        metric="sqeuclidean",
        dtype=dtype,
        add_data_on_build=False,
        array_type=array_type,
    )


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
@pytest.mark.parametrize("array_type", ["host", "device"])
def test_ivf_pq_dtype(inplace, dtype, array_type):
    run_ivf_pq_build_search_test(
        dtype=dtype,
        inplace=inplace,
        array_type=array_type,
    )


@pytest.mark.parametrize("force_random_rotation", [True, False])
@pytest.mark.parametrize("codebook_kind", ["subspace", "cluster"])
@pytest.mark.parametrize("n_cols", [10, 32, 63, 200])
@pytest.mark.parametrize("pq_bits", [4, 8])
@pytest.mark.parametrize("n_lists", [10, 30, 100])
def test_build_from_args(
    force_random_rotation, codebook_kind, n_cols, pq_bits, n_lists
):
    dtype = np.float32
    metric = "sqeuclidean"
    n_rows = 10000
    n_queries = 100
    k = 10
    build_params = ivf_pq.IndexParams(
        n_lists=n_lists,
        metric=metric,
        force_random_rotation=force_random_rotation,
        codebook_kind=codebook_kind,
        pq_bits=pq_bits,
    )
    dataset = generate_data((n_rows, n_cols), dtype)
    queries = generate_data((n_queries, n_cols), dtype)
    if metric == "inner_product":
        dataset = normalize(dataset, norm="l2", axis=1)
    index = ivf_pq.build(build_params, dataset)
    rotation_matrix = index.rotation_matrix
    codebook = index.codebook
    centers = cp.array(
        index.centers.copy_to_host()
    )  # TODO: Investigate stride issue for centers
    if codebook_kind == "subspace":
        assert codebook.shape[0] == index.pq_dim
    else:
        assert codebook.shape[0] == index.n_lists
    assert index.pq_len == codebook.shape[1]
    inplace = (index.pq_dim * index.pq_len) == n_cols
    if force_random_rotation or (not inplace):
        index2 = ivf_pq.build_from_args(
            build_params, n_cols, codebook, centers, rotation_matrix
        )
    else:
        index2 = ivf_pq.build_from_args(
            build_params, n_cols, codebook, centers
        )
    assert index2.codebook.shape == codebook.shape
    assert index2.centers.shape == centers.shape
    assert index2.pq_dim == index.pq_dim
    assert index2.pq_len == index.pq_len
    assert index2.pq_book_size == index.pq_book_size
    assert np.allclose(
        index2.rotation_matrix.copy_to_host(), rotation_matrix.copy_to_host()
    )
    assert np.allclose(
        cp.array(index2.codebook).get(), cp.array(codebook).get()
    )
    assert np.allclose(index2.centers.copy_to_host(), centers.get())

    indices = np.arange(n_rows, dtype=np.int64)
    index2 = ivf_pq.extend(index2, dataset, indices)

    search_params = ivf_pq.SearchParams(
        n_probes=10,
    )
    ret_output = ivf_pq.search(
        search_params, index, device_ndarray(queries), k
    )
    res_output2 = ivf_pq.search(
        search_params, index2, device_ndarray(queries), k
    )
    out_dist_device, out_idx_device = ret_output
    out_dist_device2, out_idx_device2 = res_output2
    out_dist = out_dist_device.copy_to_host()
    out_idx = out_idx_device.copy_to_host()
    out_dist2 = out_dist_device2.copy_to_host()
    out_idx2 = out_idx_device2.copy_to_host()
    tol = 1e-5
    assert np.allclose(out_dist, out_dist2, rtol=tol)
    assert (1 - np.isclose(out_idx, out_idx2).sum()) / out_idx.size < tol
