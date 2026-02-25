# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.cluster.kmeans import MiniBatchKMeans
from sklearn.cluster import MiniBatchKMeans as SklearnMiniBatchKMeans


@pytest.mark.parametrize("n_rows", [1000])
@pytest.mark.parametrize("n_cols", [10])
@pytest.mark.parametrize("n_clusters", [8])
@pytest.mark.parametrize("dtype", [np.float32])
def test_minibatch_kmeans_vs_sklearn(n_rows, n_cols, n_clusters, dtype):
    """
    Test that MiniBatchKMeans matches sklearn's MiniBatchKMeans implementation.
    """
    rng = np.random.default_rng(99)
    X_host = rng.random((n_rows, n_cols)).astype(dtype)
    initial_centroids_host = X_host[:n_clusters].copy()

    # Sklearn fit
    kmeans = SklearnMiniBatchKMeans(
        n_clusters=8,
        init=initial_centroids_host,
        max_iter=100,
        verbose=0,
        random_state=None,
        tol=0.0,
        max_no_improvement=10,
        init_size=None,
        n_init="auto",
        reassignment_ratio=0.01,
        batch_size=256,
    )
    kmeans.fit(X_host)

    centroids_sklearn = kmeans.cluster_centers_
    inertia_sklearn = kmeans.inertia_

    # cuvs MiniBatchKMeans
    mbk = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=256,
        max_iter=100,
        tol=1e-4,
        max_no_improvement=10,
        reassignment_ratio=0.01,
        init_method="Array",
    )
    mbk.fit(
        X_host,
        centroids=device_ndarray(initial_centroids_host.copy()),
    )
    centroids_cuvs = mbk.cluster_centers_.copy_to_host()
    inertia_cuvs = mbk.inertia_

    assert np.allclose(
        centroids_sklearn, centroids_cuvs, rtol=0.3, atol=0.3
    ), f"max diff: {np.max(np.abs(centroids_sklearn - centroids_cuvs))}"

    inertia_diff = abs(inertia_sklearn - inertia_cuvs)
    assert np.allclose(inertia_sklearn, inertia_cuvs, rtol=0.1, atol=0.1), (
        f"inertia diff: sklearn={inertia_sklearn}, cuvs={inertia_cuvs}, diff={inertia_diff}"
    )
