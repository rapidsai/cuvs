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

import cupy
import numpy as np
import pytest
from ann_utils import calc_recall, generate_data
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from cuvs.neighbors import cagra, hnsw

# import rmm


def run_hnsw_build_search_test(
    n_rows=10000,
    n_cols=1,
    n_queries=5,
    k=10,
    dtype=np.float32,
    metric="sqeuclidean",
    build_algo="ivf_pq",
    intermediate_graph_degree=128,
    graph_degree=64,
    search_params1={},
):
    # for i in range(2):
    # rmm.reinitialize(
    #     managed_memory=False,
    #     pool_allocator=False,
    #     initial_pool_size=2 << 30,
    # )
    dataset = generate_data((n_rows, n_cols), dtype)
    if metric == "inner_product":
        dataset = normalize(dataset, norm="l2", axis=1)
        if dtype in [np.int8, np.uint8]:
            pytest.skip(
                "inner_product metric is not supported for int8/uint8 data"
            )
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

    hnsw_index = hnsw.from_cagra(index)

    queries = generate_data((n_queries, n_cols), dtype)
    print("queries", queries)

    search_params = hnsw.SearchParams(**search_params1)

    out_dist, out_idx = hnsw.search(search_params, hnsw_index, queries, k)
    print("out_idx", cupy.asarray(out_idx))
    print("out_dist", cupy.asarray(out_dist))

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
    print("skl_idx", skl_idx)
    print("skl_dist", skl_dist)

    recall = calc_recall(out_idx, skl_idx)
    print("recall", recall)
    assert recall > 0.95
    del index


@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize("k", [3])
@pytest.mark.parametrize("ef", [30, 40])
@pytest.mark.parametrize("num_threads", [2, 4])
@pytest.mark.parametrize("metric", ["sqeuclidean"])
@pytest.mark.parametrize("build_algo", ["ivf_pq", "nn_descent"])
def test_hnsw(dtype, k, ef, num_threads, metric, build_algo):
    # Note that inner_product tests use normalized input which we cannot
    # represent in int8, therefore we test only sqeuclidean metric here.
    run_hnsw_build_search_test(
        dtype=dtype,
        k=k,
        metric=metric,
        build_algo=build_algo,
        search_params1={"ef": ef, "num_threads": num_threads},
    )
