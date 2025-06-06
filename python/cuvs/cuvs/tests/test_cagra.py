# Copyright (c) 2024, NVIDIA CORPORATION.
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
#

import numpy as np
import pytest
from pylibraft.common import device_ndarray
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from cuvs.neighbors import cagra, ivf_pq
from cuvs.tests.ann_utils import (
    calc_recall,
    generate_data,
    run_filtered_search_test,
)


def run_cagra_build_search_test(
    n_rows=10000,
    n_cols=10,
    n_queries=100,
    k=10,
    dtype=np.float32,
    metric="sqeuclidean",
    intermediate_graph_degree=128,
    graph_degree=64,
    build_algo="ivf_pq",
    array_type="device",
    compare=True,
    inplace=True,
    add_data_on_build=True,
    search_params={},
    compression=None,
):
    dataset = generate_data((n_rows, n_cols), dtype)
    if metric == "inner_product":
        if dtype in [np.int8, np.uint8]:
            pytest.skip("skip normalization for int8/uint8 data")
        dataset = normalize(dataset, norm="l2", axis=1)
    dataset_device = device_ndarray(dataset)

    build_params = cagra.IndexParams(
        metric=metric,
        intermediate_graph_degree=intermediate_graph_degree,
        graph_degree=graph_degree,
        build_algo=build_algo,
        compression=compression,
    )

    if array_type == "device":
        index = cagra.build(build_params, dataset_device)
    else:
        index = cagra.build(build_params, dataset)

    if not add_data_on_build:
        dataset_1 = dataset[: n_rows // 2, :]
        dataset_2 = dataset[n_rows // 2 :, :]
        indices_1 = np.arange(n_rows // 2, dtype=np.uint32)
        indices_2 = np.arange(n_rows // 2, n_rows, dtype=np.uint32)
        if array_type == "device":
            dataset_1_device = device_ndarray(dataset_1)
            dataset_2_device = device_ndarray(dataset_2)
            indices_1_device = device_ndarray(indices_1)
            indices_2_device = device_ndarray(indices_2)
            index = cagra.extend(index, dataset_1_device, indices_1_device)
            index = cagra.extend(index, dataset_2_device, indices_2_device)
        else:
            index = cagra.extend(index, dataset_1, indices_1)
            index = cagra.extend(index, dataset_2, indices_2)

    queries = generate_data((n_queries, n_cols), dtype)
    out_idx = np.zeros((n_queries, k), dtype=np.uint32)
    out_dist = np.zeros((n_queries, k), dtype=np.float32)

    queries_device = device_ndarray(queries)
    out_idx_device = device_ndarray(out_idx) if inplace else None
    out_dist_device = device_ndarray(out_dist) if inplace else None

    search_params = cagra.SearchParams(**search_params)

    ret_output = cagra.search(
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
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
@pytest.mark.parametrize("array_type", ["device", "host"])
@pytest.mark.parametrize("build_algo", ["ivf_pq", "nn_descent"])
@pytest.mark.parametrize("metric", ["sqeuclidean", "inner_product"])
def test_cagra_dataset_dtype_host_device(
    dtype, array_type, inplace, build_algo, metric
):
    # Note that inner_product tests use normalized input which we cannot
    # represent in int8, therefore we test only sqeuclidean metric here.
    run_cagra_build_search_test(
        dtype=dtype,
        inplace=inplace,
        array_type=array_type,
        build_algo=build_algo,
        metric=metric,
    )


@pytest.mark.parametrize("sparsity", [0.2, 0.5, 0.7, 1.0])
def test_filtered_cagra(sparsity):
    run_filtered_search_test(cagra, sparsity)


@pytest.mark.parametrize(
    "params",
    [
        {
            "intermediate_graph_degree": 64,
            "graph_degree": 32,
            "add_data_on_build": True,
            "k": 1,
            "metric": "sqeuclidean",
            "build_algo": "ivf_pq",
        },
        {
            "intermediate_graph_degree": 32,
            "graph_degree": 16,
            "add_data_on_build": False,
            "k": 5,
            "metric": "sqeuclidean",
            "build_algo": "ivf_pq",
        },
        {
            "intermediate_graph_degree": 128,
            "graph_degree": 32,
            "add_data_on_build": True,
            "k": 10,
            "metric": "inner_product",
            "build_algo": "nn_descent",
        },
    ],
)
def test_cagra_index_params(params):
    # Note that inner_product tests use normalized input which we cannot
    # represent in int8, therefore we test only sqeuclidean metric here.
    run_cagra_build_search_test(
        k=params["k"],
        metric=params["metric"],
        graph_degree=params["graph_degree"],
        intermediate_graph_degree=params["intermediate_graph_degree"],
        compare=False,
        build_algo=params["build_algo"],
    )


def test_cagra_vpq_compression():
    dim = 64
    pq_len = 2
    run_cagra_build_search_test(
        n_cols=dim, compression=cagra.CompressionParams(pq_dim=dim / pq_len)
    )


@pytest.mark.parametrize("internal_dtype", [np.float32, np.float16, np.uint8])
def test_cagra_ivf_pq(
    internal_dtype,
    n_rows=1000,
    n_cols=30,
    n_queries=20,
    k=5,
    dtype=np.float16,
    metric="inner_product",
    intermediate_graph_degree=32,
    graph_degree=16,
    build_algo="ivf_pq",
):
    dataset = generate_data((n_rows, n_cols), dtype)
    dataset = normalize(dataset, norm="l2", axis=1)
    dataset_device = device_ndarray(dataset)

    ivf_pq_params_build = ivf_pq.IndexParams(metric=metric, n_lists=10)
    ivf_pq_params_search = ivf_pq.SearchParams(
        n_probes=5,
        lut_dtype=internal_dtype,
        coarse_search_dtype=np.int8,
    )
    build_params = cagra.IndexParams(
        metric=metric,
        intermediate_graph_degree=intermediate_graph_degree,
        graph_degree=graph_degree,
        build_algo=build_algo,
        ivf_pq_build_params=ivf_pq_params_build,
        ivf_pq_search_params=ivf_pq_params_search,
        refinement_rate=1.2,
    )
    cudadtype_to_np = {np.float32: 0, np.float16: 2, np.int8: 3, np.uint8: 8}

    assert (
        build_params.ivf_pq_search_params.lut_dtype
        == cudadtype_to_np[internal_dtype]
    )
    assert (
        build_params.ivf_pq_search_params.coarse_search_dtype
        == cudadtype_to_np[np.int8]
    )
    assert np.isclose(build_params.refinement_rate, 1.2)
    index = cagra.build(build_params, dataset_device)

    queries = generate_data((n_queries, n_cols), dtype)
    queries_device = device_ndarray(queries)
    out_idx = np.zeros((n_queries, k), dtype=np.uint32)
    out_dist = np.zeros((n_queries, k), dtype=np.float32)
    out_idx_device = device_ndarray(out_idx)
    out_dist_device = device_ndarray(out_dist)

    cagra.search(
        cagra.SearchParams(),
        index,
        queries_device,
        k,
        neighbors=out_idx_device,
        distances=out_dist_device,
    )
    out_idx = out_idx_device.copy_to_host()

    skl_idx = (
        NearestNeighbors(n_neighbors=k, algorithm="brute", metric="cosine")
        .fit(dataset)
        .kneighbors(queries, return_distance=False)
    )

    recall = calc_recall(out_idx, skl_idx)

    assert recall > 0.9
