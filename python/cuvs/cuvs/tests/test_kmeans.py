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
    fit_batched,
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


@pytest.mark.parametrize("n_rows", [1000])
@pytest.mark.parametrize("n_cols", [10, 50])
@pytest.mark.parametrize("n_clusters", [5, 20])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fit_batched_fullbatch(n_rows, n_cols, n_clusters, dtype):
    """
    Test that fit_batched in FullBatch mode produces centroids that reduce
    inertia compared to the initial centroids.
    """
    rng = np.random.default_rng(42)
    X = rng.random((n_rows, n_cols)).astype(dtype)

    initial_centroids = device_ndarray(X[:n_clusters].copy())
    X_device = device_ndarray(X)
    original_inertia = cluster_cost(X_device, initial_centroids)

    params = KMeansParams(
        n_clusters=n_clusters,
        init_method="Array",
        max_iter=50,
    )

    centroids, inertia, n_iter = fit_batched(
        params, X, batch_size=256, centroids=initial_centroids
    )
    assert n_iter >= 1

    fitted_inertia = cluster_cost(X_device, centroids)
    assert fitted_inertia < original_inertia


@pytest.mark.parametrize("n_rows", [1000])
@pytest.mark.parametrize("n_cols", [10])
@pytest.mark.parametrize("n_clusters", [8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fit_batched_minibatch(n_rows, n_cols, n_clusters, dtype):
    """
    Test that fit_batched in MiniBatch mode converges (reduces inertia).
    """
    rng = np.random.default_rng(123)
    X = rng.random((n_rows, n_cols)).astype(dtype)

    initial_centroids = device_ndarray(X[:n_clusters].copy())
    X_device = device_ndarray(X)
    original_inertia = cluster_cost(X_device, initial_centroids)

    params = KMeansParams(
        n_clusters=n_clusters,
        init_method="Array",
        max_iter=200,
        update_mode="mini_batch",
    )

    centroids, inertia, n_iter = fit_batched(
        params, X, batch_size=128, centroids=initial_centroids
    )
    assert n_iter >= 1

    fitted_inertia = cluster_cost(X_device, centroids)
    assert fitted_inertia < original_inertia


@pytest.mark.parametrize("n_rows", [1000])
@pytest.mark.parametrize("n_cols", [10])
@pytest.mark.parametrize("n_clusters", [8])
@pytest.mark.parametrize("dtype", [np.float32])
def test_fit_batched_matches_fit(n_rows, n_cols, n_clusters, dtype):
    """
    Test that fit_batched FullBatch produces the same centroids as regular fit
    when given the same initial centroids.
    """
    rng = np.random.default_rng(99)
    X_host = rng.random((n_rows, n_cols)).astype(dtype)
    initial_centroids_host = X_host[:n_clusters].copy()

    # Regular fit (device data)
    params = KMeansParams(
        n_clusters=n_clusters,
        init_method="Array",
        max_iter=20,
        tol=1e-10,
    )
    centroids_regular, _, _ = fit(
        params,
        device_ndarray(X_host),
        device_ndarray(initial_centroids_host.copy()),
    )
    centroids_regular = centroids_regular.copy_to_host()

    # Batched fit (host data, full batch mode)
    centroids_batched, _, _ = fit_batched(
        params,
        X_host,
        batch_size=256,
        centroids=device_ndarray(initial_centroids_host.copy()),
    )
    centroids_batched = centroids_batched.copy_to_host()

    assert np.allclose(
        centroids_regular, centroids_batched, rtol=1e-4, atol=1e-4
    ), f"max diff: {np.max(np.abs(centroids_regular - centroids_batched))}"


@pytest.mark.parametrize("n_rows", [500])
@pytest.mark.parametrize("n_cols", [10])
@pytest.mark.parametrize("n_clusters", [5])
@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize("batch_size", [64, 128, 256, 500])
def test_fit_batched_batch_size_determinism(
    n_rows, n_cols, n_clusters, dtype, batch_size
):
    """
    Test that fit_batched FullBatch produces identical centroids regardless
    of batch_size, since the full dataset is accumulated before updating.
    """
    rng = np.random.default_rng(77)
    X = rng.random((n_rows, n_cols)).astype(dtype)
    initial_centroids_host = X[:n_clusters].copy()

    params = KMeansParams(
        n_clusters=n_clusters,
        init_method="Array",
        max_iter=20,
        tol=1e-10,
    )

    # Reference: batch_size = full dataset
    centroids_ref, _, _ = fit_batched(
        params,
        X,
        batch_size=n_rows,
        centroids=device_ndarray(initial_centroids_host.copy()),
    )
    centroids_ref = centroids_ref.copy_to_host()

    centroids_test, _, _ = fit_batched(
        params,
        X,
        batch_size=batch_size,
        centroids=device_ndarray(initial_centroids_host.copy()),
    )
    centroids_test = centroids_test.copy_to_host()

    assert np.allclose(centroids_ref, centroids_test, rtol=1e-5, atol=1e-5), (
        f"batch_size={batch_size}: max diff="
        f"{np.max(np.abs(centroids_ref - centroids_test))}"
    )


@pytest.mark.parametrize("n_rows", [1000])
@pytest.mark.parametrize("n_cols", [10])
@pytest.mark.parametrize("n_clusters", [8])
@pytest.mark.parametrize("dtype", [np.float32])
def test_fit_batched_auto_init(n_rows, n_cols, n_clusters, dtype):
    """
    Test fit_batched without providing initial centroids (auto-initialization).
    """
    rng = np.random.default_rng(55)
    X = rng.random((n_rows, n_cols)).astype(dtype)

    params = KMeansParams(n_clusters=n_clusters, max_iter=50)

    centroids, inertia, n_iter = fit_batched(params, X, batch_size=256)
    assert centroids.shape == (n_clusters, n_cols)
    assert n_iter >= 1


@pytest.mark.parametrize("n_rows", [500])
@pytest.mark.parametrize("n_cols", [10])
@pytest.mark.parametrize("n_clusters", [5])
@pytest.mark.parametrize("dtype", [np.float32])
def test_fit_batched_with_sample_weights(n_rows, n_cols, n_clusters, dtype):
    """
    Test that fit_batched accepts and runs with sample weights.
    """
    rng = np.random.default_rng(66)
    X = rng.random((n_rows, n_cols)).astype(dtype)
    weights = np.ones(n_rows, dtype=dtype)

    initial_centroids = device_ndarray(X[:n_clusters].copy())
    X_device = device_ndarray(X)
    original_inertia = cluster_cost(X_device, initial_centroids)

    params = KMeansParams(
        n_clusters=n_clusters,
        init_method="Array",
        max_iter=50,
    )

    centroids, inertia, n_iter = fit_batched(
        params,
        X,
        batch_size=128,
        centroids=initial_centroids,
        sample_weights=weights,
    )

    fitted_inertia = cluster_cost(X_device, centroids)
    assert fitted_inertia < original_inertia
