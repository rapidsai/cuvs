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

from cuvs.common import Resources, SNMGResources
from cuvs.neighbors import all_neighbors, brute_force, ivf_pq, nn_descent
from cuvs.tests.ann_utils import calc_recall

cupy = pytest.importorskip("cupy")


@pytest.mark.parametrize("algo", ["nn_descent", "brute_force"])
@pytest.mark.parametrize("metric", ["sqeuclidean", "cosine"])
def test_all_neighbors_device_build_quality(algo, metric):
    """Test device build with quality validation against brute force ground
    truth.
    """
    n_rows, n_cols, k = 7151, 64, 16

    np.random.seed(42)
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(
        n_samples=n_rows,
        n_features=n_cols,
        centers=10,
        cluster_std=1.0,
        center_box=(-10.0, 10.0),
        random_state=42,
    )
    X = X.astype(np.float32)
    X_device = device_ndarray(X)

    nn_descent_params = None
    if algo == "nn_descent":
        nn_descent_params = nn_descent.IndexParams(
            metric=metric,
            graph_degree=k,
            intermediate_graph_degree=k * 2,
            max_iterations=100,
            termination_threshold=0.001,
        )

    params = all_neighbors.AllNeighborsParams(
        algo=algo,
        overlap_factor=0,
        n_clusters=1,
        metric=metric,
        nn_descent_params=nn_descent_params,
    )

    res = Resources()
    indices, distances = all_neighbors.build(
        X_device,
        k,
        params,
        distances=cupy.empty((n_rows, k), dtype=cupy.float32),
        resources=res,
    )

    bf_index = brute_force.build(X_device, metric=metric)
    bf_distances, bf_indices = brute_force.search(bf_index, X_device, k=k)

    indices_host = cupy.asnumpy(indices)
    bf_indices_host = cupy.asnumpy(bf_indices)

    assert indices.shape == (n_rows, k)
    assert indices.dtype == cupy.int64
    assert distances.shape == (n_rows, k)
    assert distances.dtype == cupy.float32

    recall = calc_recall(indices_host, bf_indices_host)
    assert recall > 0.9


@pytest.mark.parametrize("algo", ["nn_descent", "brute_force", "ivf_pq"])
def test_all_neighbors_host_build_quality(algo):
    """Test host build with quality validation against brute force ground
    truth.
    """
    n_rows, n_cols, k = 7151, 64, 16
    n_clusters = 1

    np.random.seed(42)
    from sklearn.datasets import make_blobs

    X_host, _ = make_blobs(
        n_samples=n_rows,
        n_features=n_cols,
        centers=10,
        cluster_std=1.0,
        center_box=(-10.0, 10.0),
        random_state=42,
    )
    X_host = X_host.astype(np.float32)
    X_device = device_ndarray(X_host)

    overlap_factor = 0

    ivf_pq_params = None
    nn_descent_params = None

    if algo == "ivf_pq":
        ivf_pq_params = ivf_pq.IndexParams(
            metric="sqeuclidean",
            n_lists=32,
            pq_bits=8,
            pq_dim=0,
            add_data_on_build=True,
        )
    elif algo == "nn_descent":
        nn_descent_params = nn_descent.IndexParams(
            metric="sqeuclidean",
            graph_degree=k,
            intermediate_graph_degree=k * 2,
            max_iterations=100,
            termination_threshold=0.001,
        )

    params = all_neighbors.AllNeighborsParams(
        algo=algo,
        overlap_factor=overlap_factor,
        n_clusters=n_clusters,
        metric="sqeuclidean",
        ivf_pq_params=ivf_pq_params,
        nn_descent_params=nn_descent_params,
    )

    res = SNMGResources()
    indices, distances = all_neighbors.build(
        X_host,
        k,
        params,
        distances=cupy.empty((n_rows, k), dtype=cupy.float32),
        resources=res,
    )

    bf_index = brute_force.build(X_device, metric="sqeuclidean")
    bf_distances, bf_indices = brute_force.search(bf_index, X_device, k=k)

    indices_host = cupy.asnumpy(indices)
    bf_indices_host = cupy.asnumpy(bf_indices)

    assert indices.shape == (n_rows, k)
    assert indices.dtype == cupy.int64
    assert distances.shape == (n_rows, k)
    assert distances.dtype == cupy.float32

    recall = calc_recall(indices_host, bf_indices_host)
    assert recall > 0.9
