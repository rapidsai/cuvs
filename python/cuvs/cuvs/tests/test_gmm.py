# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from pylibraft.common import device_ndarray
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture as skGaussianMixture

from cuvs.cluster.gmm import (
    GMMParams,
    fit,
    predict,
    predict_proba,
    score_samples,
)

COVARIANCE_TYPES = ["full", "tied", "diag", "spherical"]
INIT_METHODS = ["kmeans", "k-means++", "random", "random_from_data"]
DTYPES = [np.float32, np.float64]


def _blobs(n_samples=1000, n_features=8, centers=4, seed=0, dtype=np.float32):
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=1.0,
        random_state=seed,
    )
    return np.ascontiguousarray(X.astype(dtype)), y


def _rel(dtype):
    return 1e-2 if dtype == np.float32 else 1e-5


@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fit_matches_sklearn(covariance_type, dtype):
    X, y = _blobs(dtype=dtype)
    Xd = device_ndarray(X)

    params = GMMParams(
        n_components=4,
        covariance_type=covariance_type,
        init_method="kmeans",
        seed=0,
    )
    out = fit(params, Xd)
    weights, means, prec_chol = out[0], out[1], out[3]
    n_iter, converged = out[7], out[8]

    assert n_iter >= 1
    # well-separated blobs with default max_iter converge
    assert converged

    labels = predict(params, Xd, weights, means, prec_chol).copy_to_host()
    score = float(
        np.asarray(
            score_samples(params, Xd, weights, means, prec_chol).copy_to_host()
        ).mean()
    )

    sk = skGaussianMixture(
        n_components=4,
        covariance_type=covariance_type,
        init_params="kmeans",
        random_state=0,
    ).fit(X)

    # Hard labels recover the ground-truth blobs and agree with sklearn.
    assert adjusted_rand_score(y, labels) >= 0.95
    assert adjusted_rand_score(sk.predict(X), labels) >= 0.95
    # Per-sample average log-likelihood matches sklearn's GMM.score.
    assert score == pytest.approx(sk.score(X), rel=_rel(dtype))


@pytest.mark.parametrize("init_method", INIT_METHODS)
def test_init_methods_run(init_method):
    X, y = _blobs()
    Xd = device_ndarray(X)
    params = GMMParams(n_components=4, init_method=init_method, seed=0)
    out = fit(params, Xd)
    labels = predict(params, Xd, out[0], out[1], out[3]).copy_to_host()
    # kmeans-family inits recover the clusters; random inits at least run and
    # return a valid labeling.
    if init_method in ("kmeans", "k-means++"):
        assert adjusted_rand_score(y, labels) >= 0.95
    assert set(np.unique(labels)).issubset(set(range(4)))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
def test_predict_proba_normalized(covariance_type, dtype):
    X, _ = _blobs(dtype=dtype)
    Xd = device_ndarray(X)
    params = GMMParams(n_components=4, covariance_type=covariance_type, seed=0)
    out = fit(params, Xd)
    resp = predict_proba(params, Xd, out[0], out[1], out[3]).copy_to_host()
    assert resp.shape == (X.shape[0], 4)
    assert np.all(resp >= -1e-5)
    np.testing.assert_allclose(resp.sum(axis=1), 1.0, atol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
def test_predict_matches_predict_proba_argmax(covariance_type, dtype):
    X, _ = _blobs(dtype=dtype)
    Xd = device_ndarray(X)
    params = GMMParams(n_components=4, covariance_type=covariance_type, seed=0)
    out = fit(params, Xd)
    labels = predict(params, Xd, out[0], out[1], out[3]).copy_to_host()
    resp = predict_proba(params, Xd, out[0], out[1], out[3]).copy_to_host()
    np.testing.assert_array_equal(labels, resp.argmax(axis=1))


def test_score_samples_matches_sklearn():
    X, _ = _blobs(dtype=np.float64)
    Xd = device_ndarray(X)
    params = GMMParams(
        n_components=4, covariance_type="full", init_method="kmeans", seed=0
    )
    out = fit(params, Xd)
    logp = score_samples(params, Xd, out[0], out[1], out[3]).copy_to_host()

    sk = skGaussianMixture(
        n_components=4,
        covariance_type="full",
        init_params="kmeans",
        random_state=0,
    ).fit(X)
    np.testing.assert_allclose(
        np.asarray(logp), sk.score_samples(X), rtol=1e-4, atol=1e-4
    )


def test_warm_start():
    X, _ = _blobs()
    Xd = device_ndarray(X)
    params = GMMParams(
        n_components=4,
        covariance_type="full",
        init_method="kmeans",
        max_iter=1,
        seed=0,
    )
    out = fit(params, Xd)
    # Re-fit warm-started from the previous parameters; should run, return a
    # finite lower bound, and not regress the objective below the prior fit.
    out2 = fit(
        params,
        Xd,
        weights=out[0],
        means=out[1],
        covariances=out[2],
        warm_start=True,
    )
    assert np.isfinite(out2[6])
    assert out2[6] >= out[6] - 1e-6
