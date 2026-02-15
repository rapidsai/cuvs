# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.cluster.kmeans import KMeansParams, cluster_cost, fit, predict
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
@pytest.mark.parametrize("n_cols", [10])
@pytest.mark.parametrize("n_clusters", [8])
@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize(
    "batch_samples_list",
    [
        [32, 64, 128, 256, 512],  # various batch sizes
    ],
)
def test_kmeans_batch_size_determinism(
    n_rows, n_cols, n_clusters, dtype, batch_samples_list
):
    """
    Test that different batch sizes produce identical centroids.

    When starting from the same initial centroids, the k-means algorithm
    should produce identical final centroids regardless of the batch_samples
    parameter. This is because the accumulated adjustments to centroids after
    the entire dataset pass should be the same.
    """
    # Use fixed seed for reproducibility
    rng = np.random.default_rng(42)

    # Generate random data
    X_host = rng.random((n_rows, n_cols)).astype(dtype)
    X = device_ndarray(X_host)

    # Generate fixed initial centroids (using first n_clusters rows)
    initial_centroids_host = X_host[:n_clusters].copy()

    # Store results from each batch size
    results = []

    for batch_samples in batch_samples_list:
        # Create fresh copy of initial centroids for each run
        centroids = device_ndarray(initial_centroids_host.copy())

        params = KMeansParams(
            n_clusters=n_clusters,
            init_method="Array",  # Use provided centroids
            max_iter=100,
            tol=1e-10,  # Very small tolerance to ensure convergence
            batch_samples=batch_samples,
        )

        centroids_out, inertia, n_iter = fit(params, X, centroids)
        results.append(
            {
                "batch_samples": batch_samples,
                "centroids": centroids_out.copy_to_host(),
                "inertia": inertia,
                "n_iter": n_iter,
            }
        )

    # Compare all results against the first one
    reference = results[0]
    for result in results[1:]:
        # Centroids should be identical (or very close due to float precision)
        assert np.allclose(
            reference["centroids"],
            result["centroids"],
            rtol=1e-5,
            atol=1e-5,
        ), (
            f"Centroids differ between batch_samples="
            f"{reference['batch_samples']} and {result['batch_samples']}"
        )

        # Inertia should also be identical
        assert np.allclose(
            reference["inertia"], result["inertia"], rtol=1e-5, atol=1e-5
        ), (
            f"Inertia differs between batch_samples="
            f"{reference['batch_samples']} and {result['batch_samples']}: "
            f"{reference['inertia']} vs {result['inertia']}"
        )
