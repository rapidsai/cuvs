# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import pytest

from cuvs.preprocessing import pca


@pytest.mark.parametrize("n_rows", [256, 512])
@pytest.mark.parametrize("n_cols", [32, 64])
def test_fit_transform_inverse_transform(n_rows, n_cols):
    """
    fit_transform with all components then inverse_transform
    should reconstruct the original data near-losslessly.
    """
    n_components = n_cols
    X = cp.random.random_sample((n_rows, n_cols), dtype=cp.float32)

    params = pca.Params(n_components=n_components)
    result = pca.fit_transform(params, X)

    assert result.trans_input.shape == (n_rows, n_components)
    assert result.components.shape == (n_components, n_cols)

    reconstructed = pca.inverse_transform(
        params,
        result.trans_input,
        result.components,
        result.singular_vals,
        result.mu,
    )

    max_err = float(cp.max(cp.abs(cp.asfortranarray(X) - reconstructed)))
    assert max_err < 1e-3, (
        f"Reconstruction error {max_err} too large for lossless case"
    )


@pytest.mark.parametrize("n_rows", [256, 512])
@pytest.mark.parametrize("n_cols", [32, 64])
def test_fit_then_transform(n_rows, n_cols):
    """
    fit() then transform() separately should give the same result
    as fit_transform() when copy=True.
    """
    n_components = n_cols
    X = cp.random.random_sample((n_rows, n_cols), dtype=cp.float32)

    params = pca.Params(n_components=n_components, copy=True)
    fit_result = pca.fit(params, X)

    assert fit_result.components.shape == (n_components, n_cols)
    assert fit_result.singular_vals.shape == (n_components,)
    assert fit_result.mu.shape == (n_cols,)

    transformed = pca.transform(
        params,
        X,
        fit_result.components,
        fit_result.singular_vals,
        fit_result.mu,
    )
    assert transformed.shape == (n_rows, n_components)

    reconstructed = pca.inverse_transform(
        params,
        transformed,
        fit_result.components,
        fit_result.singular_vals,
        fit_result.mu,
    )

    max_err = float(cp.max(cp.abs(cp.asfortranarray(X) - reconstructed)))
    assert max_err < 1e-3, (
        f"Reconstruction error {max_err} too large for lossless case"
    )


@pytest.mark.parametrize("n_rows", [512])
@pytest.mark.parametrize("n_cols", [64])
@pytest.mark.parametrize("n_components", [8, 16])
def test_dim_reduction(n_rows, n_cols, n_components):
    """With fewer components, reconstruction should have bounded error."""
    X = cp.random.random_sample((n_rows, n_cols), dtype=cp.float32)

    params = pca.Params(n_components=n_components)
    result = pca.fit_transform(params, X)

    assert result.trans_input.shape == (n_rows, n_components)

    reconstructed = pca.inverse_transform(
        params,
        result.trans_input,
        result.components,
        result.singular_vals,
        result.mu,
    )

    max_err = float(cp.max(cp.abs(cp.asfortranarray(X) - reconstructed)))
    assert max_err > 1e-5, (
        "Reconstruction error should be non-zero with fewer components"
    )
    assert max_err < 2.0, f"Reconstruction error {max_err} should be bounded"


def test_explained_variance():
    """
    When all components are kept, explained_var_ratio should sum
    to approximately 1.0.
    """
    n_rows, n_cols = 512, 32
    X = cp.random.random_sample((n_rows, n_cols), dtype=cp.float32)

    params = pca.Params(n_components=n_cols)
    result = pca.fit(params, X)

    ratio_sum = float(cp.sum(result.explained_var_ratio))
    assert abs(ratio_sum - 1.0) < 0.05, (
        f"Explained variance ratio sum {ratio_sum} not close to 1.0"
    )

    assert cp.all(result.explained_var >= 0), (
        "Explained variances should be non-negative"
    )
    assert cp.all(result.singular_vals >= 0), (
        "Singular values should be non-negative"
    )


def test_params_defaults():
    """Test that Params has sensible defaults."""
    p = pca.Params()
    assert p.n_components == 1
    assert p.copy is True
    assert p.whiten is False
    assert p.algorithm == "cov_eig_dq"
    assert p.tol == 0.0
    assert p.n_iterations == 15


def test_params_custom():
    """Test that custom Params values propagate correctly."""
    p = pca.Params(
        n_components=10,
        copy=False,
        whiten=True,
        algorithm="cov_eig_jacobi",
        tol=1e-4,
        n_iterations=50,
    )
    assert p.n_components == 10
    assert p.copy is False
    assert p.whiten is True
    assert p.algorithm == "cov_eig_jacobi"
    assert abs(p.tol - 1e-4) < 1e-7
    assert p.n_iterations == 50
