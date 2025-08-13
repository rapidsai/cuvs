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

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.neighbors import brute_force, nn_descent
from cuvs.tests.ann_utils import calc_recall


@pytest.mark.parametrize("n_rows", [1024, 2048])
@pytest.mark.parametrize("n_cols", [32, 64])
@pytest.mark.parametrize("device_memory", [True, False])
@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("return_distances", [True, False])
def test_nn_descent(
    n_rows, n_cols, device_memory, dtype, inplace, return_distances
):
    # because of a limitation in the c++ api, we can't both return the
    # distances and have an inplace graph
    if inplace and return_distances:
        pytest.skip("Can't return distances with an inplace graph")

    metric = "sqeuclidean"
    graph_degree = 64

    input1 = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    input1_device = device_ndarray(input1)
    graph = np.zeros((n_rows, graph_degree), dtype="uint32")

    params = nn_descent.IndexParams(
        metric=metric,
        graph_degree=graph_degree,
        return_distances=return_distances,
    )
    index = nn_descent.build(
        params,
        input1_device if device_memory else input1,
        graph=graph if inplace else None,
    )

    if not inplace:
        graph = index.graph

    bfknn_index = brute_force.build(input1_device, metric=metric)
    _, bfknn_graph = brute_force.search(
        bfknn_index, input1_device, k=graph_degree
    )
    bfknn_graph = bfknn_graph.copy_to_host()

    if return_distances:
        distances = index.distances
        assert distances.shape == graph.shape

    assert calc_recall(graph, bfknn_graph) > 0.9
