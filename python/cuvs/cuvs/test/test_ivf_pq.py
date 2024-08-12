# Copyright (c) 2024, NVIDIA CORPORATION.
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from cuvs.neighbors import ivf_pq
from cuvs.test.ann_utils import calc_recall, generate_data


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

    index = ivf_pq.build(build_params, dataset_device)
    if not add_data_on_build:
        dataset_1 = dataset[: n_rows // 2, :]
        dataset_2 = dataset[n_rows // 2 :, :]
        indices_1 = np.arange(n_rows // 2, dtype=np.int64)
        indices_2 = np.arange(n_rows // 2, n_rows, dtype=np.int64)

        dataset_1_device = device_ndarray(dataset_1)
        dataset_2_device = device_ndarray(dataset_2)
        indices_1_device = device_ndarray(indices_1)
        indices_2_device = device_ndarray(indices_2)
        index = ivf_pq.extend(index, dataset_1_device, indices_1_device)
        index = ivf_pq.extend(index, dataset_2_device, indices_2_device)

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
def test_extend(dtype):
    run_ivf_pq_build_search_test(
        n_rows=10000,
        n_cols=10,
        n_queries=100,
        k=10,
        n_lists=100,
        metric="sqeuclidean",
        dtype=dtype,
        add_data_on_build=False,
    )


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.uint8])
def test_ivf_pq_dtype(inplace, dtype):
    run_ivf_pq_build_search_test(
        dtype=dtype,
        inplace=inplace,
    )
