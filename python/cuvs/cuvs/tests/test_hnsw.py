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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from cuvs.neighbors import cagra, hnsw
from cuvs.tests.ann_utils import calc_recall, generate_data


def run_hnsw_build_search_test(
    n_rows=10000,
    n_cols=10,
    n_queries=100,
    k=10,
    dtype=np.float32,
    metric="sqeuclidean",
    build_algo="ivf_pq",
    intermediate_graph_degree=128,
    graph_degree=64,
    hierarchy="none",
    search_params={},
    expected_recall=0.9,
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
    build_params = cagra.IndexParams(
        metric=metric,
        intermediate_graph_degree=intermediate_graph_degree,
        graph_degree=graph_degree,
        build_algo=build_algo,
    )

    index = cagra.build(build_params, dataset)

    assert index.trained

    hnsw_params = hnsw.IndexParams(hierarchy=hierarchy)
    hnsw_index = hnsw.from_cagra(hnsw_params, index)

    search_params = hnsw.SearchParams(**search_params)

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
    assert recall >= expected_recall


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
@pytest.mark.parametrize("k", [10, 20])
@pytest.mark.parametrize("ef", [50, 150])
@pytest.mark.parametrize("num_threads", [2, 4])
@pytest.mark.parametrize("metric", ["sqeuclidean", "inner_product"])
@pytest.mark.parametrize("build_algo", ["ivf_pq", "nn_descent"])
@pytest.mark.parametrize("hierarchy", ["none", "cpu", "gpu"])
def test_hnsw(dtype, k, ef, num_threads, metric, build_algo, hierarchy):
    expected_recall = (
        0.9 if metric == "inner_product" and dtype == np.uint8 else 0.95
    )
    run_hnsw_build_search_test(
        dtype=dtype,
        k=k,
        metric=metric,
        build_algo=build_algo,
        hierarchy=hierarchy,
        search_params={"ef": ef, "num_threads": num_threads},
        expected_recall=expected_recall,
    )


def run_hnsw_extend_test(
    n_rows=10000,
    add_rows=2000,
    n_cols=10,
    n_queries=100,
    k=10,
    dtype=np.float32,
    metric="sqeuclidean",
    build_algo="ivf_pq",
    intermediate_graph_degree=128,
    graph_degree=64,
    search_params={},
    hierarchy="cpu",
):
    dataset = generate_data((n_rows, n_cols), dtype)
    add_dataset = generate_data((add_rows, n_cols), dtype)
    queries = generate_data((n_queries, n_cols), dtype)
    if metric == "inner_product":
        dataset = normalize(dataset, norm="l2", axis=1)
        add_dataset = normalize(add_dataset, norm="l2", axis=1)
        queries = normalize(queries, norm="l2", axis=1)
        if dtype in [np.int8, np.uint8]:
            # Quantize the normalized data to the int8/uint8 range
            dtype_max = np.iinfo(dtype).max
            dataset = (dataset * dtype_max).astype(dtype)
            add_dataset = (add_dataset * dtype_max).astype(dtype)
            queries = (queries * dtype_max).astype(dtype)
        if build_algo == "nn_descent":
            pytest.skip("inner_product metric is not supported for nn_descent")

    build_params = cagra.IndexParams(
        metric=metric,
        intermediate_graph_degree=intermediate_graph_degree,
        graph_degree=graph_degree,
        build_algo=build_algo,
    )

    index = cagra.build(build_params, dataset)

    assert index.trained

    hnsw_params = hnsw.IndexParams(hierarchy=hierarchy)
    hnsw_index = hnsw.from_cagra(hnsw_params, index)
    hnsw.extend(hnsw.ExtendParams(), hnsw_index, add_dataset)

    search_params = hnsw.SearchParams(**search_params)

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
    nn_skl.fit(np.vstack([dataset, add_dataset]))
    skl_dist, skl_idx = nn_skl.kneighbors(queries, return_distance=True)

    recall = calc_recall(out_idx, skl_idx)
    assert recall > 0.95


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.uint8])
@pytest.mark.parametrize("k", [10, 20])
@pytest.mark.parametrize("ef", [30, 40])
@pytest.mark.parametrize("num_threads", [2, 4])
@pytest.mark.parametrize("metric", ["sqeuclidean"])
@pytest.mark.parametrize("build_algo", ["ivf_pq", "nn_descent"])
@pytest.mark.parametrize("hierarchy", ["cpu", "gpu"])
def test_hnsw_extend(dtype, k, ef, num_threads, metric, build_algo, hierarchy):
    # Note that inner_product tests use normalized input which we cannot
    # represent in int8, therefore we test only sqeuclidean metric here.
    run_hnsw_extend_test(
        dtype=dtype,
        k=k,
        metric=metric,
        build_algo=build_algo,
        search_params={"ef": ef, "num_threads": num_threads},
        hierarchy=hierarchy,
    )
