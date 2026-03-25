# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.cluster.kmeans import (
    KMeansParams,
    cluster_cost,
    fit,
    fit_predict,
    predict,
)
from cuvs.distance import pairwise_distance


@pytest.mark.parametrize("n_rows", [100])
@pytest.mark.parametrize("n_cols", [5, 25])
@pytest.mark.parametrize("n_clusters", [5, 15])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("hierarchical", [True, False])
def test_kmeans_fit(n_rows, n_cols, n_clusters, dtype, hierarchical):
    if hierarchical and dtype == np.float64:
        pytest.skip("hierarchical kmeans doesn't support float64")

    # generate some random input points / centroids
    X_host = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    centroids = device_ndarray(X_host[:n_clusters])
    X = device_ndarray(X_host)

    # compute the inertia, before fitting centroids
    original_inertia = cluster_cost(X, centroids)

    params = KMeansParams(n_clusters=n_clusters, hierarchical=hierarchical)

    # fit the centroids, make sure inertia has gone down
    centroids, inertia, n_iter = fit(params, X, centroids)
    assert n_iter >= 1

    # balanced kmeans doesn't return inertia
    if not hierarchical:
        assert inertia < original_inertia
        assert np.allclose(cluster_cost(X, centroids), inertia, rtol=1e-6)

    # make sure the prediction for each centroid is the centroid itself
    labels, inertia = predict(params, centroids, centroids)
    assert np.all(labels.copy_to_host() == np.arange(labels.shape[0]))


@pytest.mark.parametrize("n_rows", [100])
@pytest.mark.parametrize("n_cols", [5, 25])
@pytest.mark.parametrize("n_clusters", [5, 15])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("hierarchical", [True, False])
def test_kmeans_fit_predict(n_rows, n_cols, n_clusters, dtype, hierarchical):
    if hierarchical and dtype == np.float64:
        pytest.skip("hierarchical kmeans doesn't support float64")

    # generate some random input points / centroids
    X_host = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    init_host = X_host[:n_clusters].copy()

    # initialize the centroids for fit_predict and sequential fit
    init_for_fit_predict = device_ndarray(init_host)
    init_for_sequential = device_ndarray(init_host.copy())

    X = device_ndarray(X_host)

    params = KMeansParams(
        n_clusters=n_clusters,
        hierarchical=hierarchical,
        init_method="Array",
    )

    labels_fp, centroids_fp, inertia_fp, n_iter_fp = fit_predict(
        params, X, centroids=init_for_fit_predict
    )
    centroids_seq, _, n_iter_seq = fit(
        params, X, centroids=init_for_sequential
    )

    if hierarchical:
        labels, pred_inertia = predict(params, X, centroids_fp)
        assert inertia_fp == pred_inertia == 0.0
    else:
        labels, _ = predict(params, X, centroids_seq)
        assert n_iter_fp == n_iter_seq

    # make sure the labels are the same between fit_predict and sequential fit
    np.testing.assert_array_equal(
        labels_fp.copy_to_host(),
        labels.copy_to_host(),
    )


@pytest.mark.parametrize("n_rows", [100])
@pytest.mark.parametrize("n_cols", [5, 25])
@pytest.mark.parametrize("n_clusters", [4, 15])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cluster_cost(n_rows, n_cols, n_clusters, dtype):
    X = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    X_device = device_ndarray(X)

    centroids = X[:n_clusters]
    centroids_device = device_ndarray(centroids)

    inertia = cluster_cost(X_device, centroids_device)

    # compute the nearest centroid to each sample
    distances = pairwise_distance(
        X_device, centroids_device, metric="sqeuclidean"
    ).copy_to_host()
    cluster_ids = np.argmin(distances, axis=1)

    cluster_distances = np.take_along_axis(
        distances, cluster_ids[:, None], axis=1
    )

    # need reduced tolerance for float32
    tol = 1e-3 if dtype == np.float32 else 1e-6
    assert np.allclose(inertia, sum(cluster_distances), rtol=tol, atol=tol)
