#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from collections import namedtuple

import cupy as cp
import numpy as np

from libcpp cimport bool

from cuvs.common cimport cydlpack

from pylibraft.common import auto_convert_output, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array

from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources
from cuvs.neighbors.common import _check_input_array

SOLVER_TYPES = {
    "cov_eig_dq": cuvsPcaSolver.CUVS_PCA_COV_EIG_DQ,
    "cov_eig_jacobi": cuvsPcaSolver.CUVS_PCA_COV_EIG_JACOBI,
}

SOLVER_NAMES = {v: k for k, v in SOLVER_TYPES.items()}


cdef class Params:
    """
    Parameters for PCA decomposition.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep (default: 1).
    copy : bool
        If False, data passed to fit are overwritten and running
        fit(X) then transform(X) will not yield the expected results;
        use fit_transform(X) instead (default: True).
    whiten : bool
        When True the component vectors are multiplied by the square root
        of n_samples and divided by the singular values to ensure
        uncorrelated outputs with unit component-wise variances
        (default: False).
    algorithm : str
        Solver algorithm. One of ``"cov_eig_dq"`` (divide-and-conquer)
        or ``"cov_eig_jacobi"`` (Jacobi) (default: ``"cov_eig_dq"``).
    tol : float
        Tolerance for singular values, used by the Jacobi solver
        (default: 0.0).
    n_iterations : int
        Number of iterations for the Jacobi solver (default: 15).
    """

    cdef cuvsPcaParams* params

    def __cinit__(self):
        check_cuvs(cuvsPcaParamsCreate(&self.params))

    def __dealloc__(self):
        check_cuvs(cuvsPcaParamsDestroy(self.params))

    def __init__(self, *, n_components=None, copy=None, whiten=None,
                 algorithm=None, tol=None, n_iterations=None):
        if n_components is not None:
            self.params.n_components = n_components
        if copy is not None:
            self.params.copy = copy
        if whiten is not None:
            self.params.whiten = whiten
        if algorithm is not None:
            self.params.algorithm = <cuvsPcaSolver>SOLVER_TYPES[algorithm]
        if tol is not None:
            self.params.tol = tol
        if n_iterations is not None:
            self.params.n_iterations = n_iterations

    @property
    def n_components(self):
        return self.params.n_components

    @property
    def copy(self):
        return self.params.copy

    @property
    def whiten(self):
        return self.params.whiten

    @property
    def algorithm(self):
        return SOLVER_NAMES[self.params.algorithm]

    @property
    def tol(self):
        return self.params.tol

    @property
    def n_iterations(self):
        return self.params.n_iterations


FitOutput = namedtuple(
    "FitOutput",
    "components explained_var explained_var_ratio "
    "singular_vals mu noise_vars",
)

FitTransformOutput = namedtuple(
    "FitTransformOutput",
    "trans_input components explained_var explained_var_ratio "
    "singular_vals mu noise_vars",
)


def _to_f_order(ary):
    """Ensure a device array is Fortran-contiguous (col-major)."""
    if hasattr(ary, "__cuda_array_interface__"):
        return cp.asfortranarray(cp.asarray(ary))
    return np.asfortranarray(np.asarray(ary))


@auto_sync_resources
def fit(Params params, X, resources=None):
    """
    Compute PCA (fit only).

    Computes the principal components, explained variances, singular
    values, and column means from the input data.

    Parameters
    ----------
    params : Params
        PCA parameters. ``params.copy`` should be True if you intend
        to reuse *X* after this call.
    X : device array-like, shape (n_samples, n_features), float32
        Input data (will be converted to col-major device memory).
    {resources_docstring}

    Returns
    -------
    FitOutput
        Named tuple with fields: ``components``, ``explained_var``,
        ``explained_var_ratio``, ``singular_vals``, ``mu``,
        ``noise_vars``.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing import pca
    >>> X = cp.random.random_sample((500, 32), dtype=cp.float32)
    >>> params = pca.Params(n_components=8, copy=True)
    >>> result = pca.fit(params, X)
    >>> result.components.shape
    (8, 32)
    """
    n_components = params.n_components

    X_f = _to_f_order(X)
    x_ai = wrap_array(X_f)
    _check_input_array(x_ai, [np.dtype("float32")])
    n_rows, n_cols = x_ai.shape

    components = cp.empty((n_components, n_cols), dtype="float32", order="F")
    explained_var = cp.empty((n_components,), dtype="float32")
    explained_var_ratio = cp.empty((n_components,), dtype="float32")
    singular_vals = cp.empty((n_components,), dtype="float32")
    mu = cp.empty((n_cols,), dtype="float32")
    noise_vars = cp.empty((1,), dtype="float32")

    cdef cydlpack.DLManagedTensor* x_dlpack = \
        cydlpack.dlpack_c(x_ai)
    cdef cydlpack.DLManagedTensor* comp_dlpack = \
        cydlpack.dlpack_c(wrap_array(components))
    cdef cydlpack.DLManagedTensor* ev_dlpack = \
        cydlpack.dlpack_c(wrap_array(explained_var))
    cdef cydlpack.DLManagedTensor* evr_dlpack = \
        cydlpack.dlpack_c(wrap_array(explained_var_ratio))
    cdef cydlpack.DLManagedTensor* sv_dlpack = \
        cydlpack.dlpack_c(wrap_array(singular_vals))
    cdef cydlpack.DLManagedTensor* mu_dlpack = \
        cydlpack.dlpack_c(wrap_array(mu))
    cdef cydlpack.DLManagedTensor* nv_dlpack = \
        cydlpack.dlpack_c(wrap_array(noise_vars))

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    check_cuvs(cuvsPcaFit(res, params.params, x_dlpack,
                           comp_dlpack, ev_dlpack, evr_dlpack,
                           sv_dlpack, mu_dlpack, nv_dlpack, False))

    return FitOutput(components, explained_var, explained_var_ratio,
                     singular_vals, mu, noise_vars)


@auto_sync_resources
def fit_transform(Params params, X, resources=None):
    """
    Compute PCA and transform the input data in a single operation.

    Parameters
    ----------
    params : Params
        PCA parameters.
    X : device array-like, shape (n_samples, n_features), float32
        Input data (will be converted to col-major device memory).
    {resources_docstring}

    Returns
    -------
    FitTransformOutput
        Named tuple with fields: ``trans_input``, ``components``,
        ``explained_var``, ``explained_var_ratio``, ``singular_vals``,
        ``mu``, ``noise_vars``.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing import pca
    >>> X = cp.random.random_sample((500, 32), dtype=cp.float32)
    >>> params = pca.Params(n_components=8)
    >>> result = pca.fit_transform(params, X)
    >>> result.trans_input.shape
    (500, 8)
    """
    n_components = params.n_components

    X_f = _to_f_order(X)
    x_ai = wrap_array(X_f)
    _check_input_array(x_ai, [np.dtype("float32")])
    n_rows, n_cols = x_ai.shape

    trans_input = cp.empty((n_rows, n_components), dtype="float32", order="F")
    components = cp.empty((n_components, n_cols), dtype="float32", order="F")
    explained_var = cp.empty((n_components,), dtype="float32")
    explained_var_ratio = cp.empty((n_components,), dtype="float32")
    singular_vals = cp.empty((n_components,), dtype="float32")
    mu = cp.empty((n_cols,), dtype="float32")
    noise_vars = cp.empty((1,), dtype="float32")

    cdef cydlpack.DLManagedTensor* x_dlpack = \
        cydlpack.dlpack_c(x_ai)
    cdef cydlpack.DLManagedTensor* trans_dlpack = \
        cydlpack.dlpack_c(wrap_array(trans_input))
    cdef cydlpack.DLManagedTensor* comp_dlpack = \
        cydlpack.dlpack_c(wrap_array(components))
    cdef cydlpack.DLManagedTensor* ev_dlpack = \
        cydlpack.dlpack_c(wrap_array(explained_var))
    cdef cydlpack.DLManagedTensor* evr_dlpack = \
        cydlpack.dlpack_c(wrap_array(explained_var_ratio))
    cdef cydlpack.DLManagedTensor* sv_dlpack = \
        cydlpack.dlpack_c(wrap_array(singular_vals))
    cdef cydlpack.DLManagedTensor* mu_dlpack = \
        cydlpack.dlpack_c(wrap_array(mu))
    cdef cydlpack.DLManagedTensor* nv_dlpack = \
        cydlpack.dlpack_c(wrap_array(noise_vars))

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    check_cuvs(cuvsPcaFitTransform(res, params.params, x_dlpack,
                                    trans_dlpack, comp_dlpack,
                                    ev_dlpack, evr_dlpack, sv_dlpack,
                                    mu_dlpack, nv_dlpack, False))

    return FitTransformOutput(trans_input, components, explained_var,
                              explained_var_ratio, singular_vals, mu,
                              noise_vars)


@auto_sync_resources
@auto_convert_output
def transform(Params params, X, components, singular_vals, mu,
              trans_input=None, resources=None):
    """
    Transform data into the PCA eigenspace.

    Uses previously computed principal components from :func:`fit` or
    :func:`fit_transform`.

    Parameters
    ----------
    params : Params
        PCA parameters (must match those used during fit).
    X : device array-like, shape (n_samples, n_features), float32
        Data to transform.
    components : device array-like, shape (n_components, n_features)
        Principal components from a prior fit.
    singular_vals : device array-like, shape (n_components,)
        Singular values from a prior fit.
    mu : device array-like, shape (n_features,)
        Column means from a prior fit.
    trans_input : optional device array, shape (n_samples, n_components)
        Pre-allocated output buffer (col-major, float32).
    {resources_docstring}

    Returns
    -------
    trans_input : device array, shape (n_samples, n_components)

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing import pca
    >>> X = cp.random.random_sample((500, 32), dtype=cp.float32)
    >>> params = pca.Params(n_components=8, copy=True)
    >>> result = pca.fit(params, X)
    >>> transformed = pca.transform(params, X, result.components,
    ...                             result.singular_vals, result.mu)
    """
    n_components = params.n_components

    X_f = _to_f_order(X)
    x_ai = wrap_array(X_f)
    _check_input_array(x_ai, [np.dtype("float32")])
    n_rows = x_ai.shape[0]

    components_f = _to_f_order(components)
    singular_vals_arr = cp.asarray(singular_vals)
    mu_arr = cp.asarray(mu)

    if trans_input is None:
        trans_input = cp.empty(
            (n_rows, n_components), dtype="float32", order="F"
        )
    else:
        trans_input = _to_f_order(trans_input)

    cdef cydlpack.DLManagedTensor* x_dlpack = \
        cydlpack.dlpack_c(x_ai)
    cdef cydlpack.DLManagedTensor* comp_dlpack = \
        cydlpack.dlpack_c(wrap_array(components_f))
    cdef cydlpack.DLManagedTensor* sv_dlpack = \
        cydlpack.dlpack_c(wrap_array(singular_vals_arr))
    cdef cydlpack.DLManagedTensor* mu_dlpack = \
        cydlpack.dlpack_c(wrap_array(mu_arr))
    cdef cydlpack.DLManagedTensor* trans_dlpack = \
        cydlpack.dlpack_c(wrap_array(trans_input))

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    check_cuvs(cuvsPcaTransform(res, params.params, x_dlpack,
                                 comp_dlpack, sv_dlpack, mu_dlpack,
                                 trans_dlpack))

    return trans_input


@auto_sync_resources
@auto_convert_output
def inverse_transform(Params params, trans_input, components,
                      singular_vals, mu, output=None, resources=None):
    """
    Transform data from the PCA eigenspace back to the original space.

    Parameters
    ----------
    params : Params
        PCA parameters (must match those used during fit).
    trans_input : device array-like, shape (n_samples, n_components)
        Transformed data from :func:`transform` or :func:`fit_transform`.
    components : device array-like, shape (n_components, n_features)
        Principal components from a prior fit.
    singular_vals : device array-like, shape (n_components,)
        Singular values from a prior fit.
    mu : device array-like, shape (n_features,)
        Column means from a prior fit.
    output : optional device array, shape (n_samples, n_features)
        Pre-allocated output buffer (col-major, float32).
    {resources_docstring}

    Returns
    -------
    output : device array, shape (n_samples, n_features)
        Reconstructed data.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.preprocessing import pca
    >>> X = cp.random.random_sample((500, 32), dtype=cp.float32)
    >>> params = pca.Params(n_components=8)
    >>> result = pca.fit_transform(params, X)
    >>> reconstructed = pca.inverse_transform(
    ...     params, result.trans_input, result.components,
    ...     result.singular_vals, result.mu)
    """
    trans_f = _to_f_order(trans_input)
    trans_ai = wrap_array(trans_f)
    _check_input_array(trans_ai, [np.dtype("float32")])
    n_rows = trans_ai.shape[0]

    components_f = _to_f_order(components)
    comp_ai = wrap_array(components_f)
    n_cols = comp_ai.shape[1]

    singular_vals_arr = cp.asarray(singular_vals)
    mu_arr = cp.asarray(mu)

    if output is None:
        output = cp.empty((n_rows, n_cols), dtype="float32", order="F")
    else:
        output = _to_f_order(output)

    cdef cydlpack.DLManagedTensor* trans_dlpack = \
        cydlpack.dlpack_c(trans_ai)
    cdef cydlpack.DLManagedTensor* comp_dlpack = \
        cydlpack.dlpack_c(comp_ai)
    cdef cydlpack.DLManagedTensor* sv_dlpack = \
        cydlpack.dlpack_c(wrap_array(singular_vals_arr))
    cdef cydlpack.DLManagedTensor* mu_dlpack = \
        cydlpack.dlpack_c(wrap_array(mu_arr))
    cdef cydlpack.DLManagedTensor* out_dlpack = \
        cydlpack.dlpack_c(wrap_array(output))

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    check_cuvs(cuvsPcaInverseTransform(res, params.params, trans_dlpack,
                                        comp_dlpack, sv_dlpack,
                                        mu_dlpack, out_dlpack))

    return output
