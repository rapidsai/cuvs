#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from collections import namedtuple

import numpy as np

cimport cuvs.common.cydlpack

from cuvs.common.resources import auto_sync_resources

from libcpp cimport bool

from cuvs.common cimport cydlpack

from pylibraft.common import auto_convert_output, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible

from cuvs.common.exceptions import check_cuvs
from cuvs.neighbors.common import _check_input_array

COVARIANCE_TYPES = {
    "full": cuvsGMMCovarianceType.CUVS_GMM_COVARIANCE_FULL,
    "tied": cuvsGMMCovarianceType.CUVS_GMM_COVARIANCE_TIED,
    "diag": cuvsGMMCovarianceType.CUVS_GMM_COVARIANCE_DIAG,
    "spherical": cuvsGMMCovarianceType.CUVS_GMM_COVARIANCE_SPHERICAL,
}

COVARIANCE_NAMES = {v: k for k, v in COVARIANCE_TYPES.items()}

INIT_METHOD_TYPES = {
    "kmeans": cuvsGMMInitMethod.CUVS_GMM_INIT_KMEANS,
    "k-means++": cuvsGMMInitMethod.CUVS_GMM_INIT_KMEANS_PLUS_PLUS,
    "random": cuvsGMMInitMethod.CUVS_GMM_INIT_RANDOM,
    "random_from_data": cuvsGMMInitMethod.CUVS_GMM_INIT_RANDOM_FROM_DATA,
}

INIT_METHOD_NAMES = {v: k for k, v in INIT_METHOD_TYPES.items()}


def _covariance_shape(covariance_type, n_components, n_features):
    """Logical shape of the covariance-typed buffers (sklearn conventions)."""
    if covariance_type == "full":
        return (n_components, n_features, n_features)
    elif covariance_type == "tied":
        return (n_features, n_features)
    elif covariance_type == "diag":
        return (n_components, n_features)
    else:  # spherical
        return (n_components,)


cdef class GMMParams:
    """
    Hyper-parameters for the Gaussian mixture EM solver

    Parameters
    ----------
    n_components : int
        The number of mixture components.
    covariance_type : str
        Covariance parameterization, one of "full", "tied", "diag",
        "spherical". Matches scikit-learn's ``GaussianMixture``.
    tol : float
        Convergence threshold on the change of the per-sample average
        log-likelihood (lower bound).
    reg_covar : float
        Non-negative regularization added to the diagonal of covariance.
    max_iter : int
        Maximum number of EM iterations for a single run.
    n_init : int
        Number of initializations to perform; the best result is kept.
    init_method : str
        Strategy used to initialize the responsibilities before EM. One of:
        "kmeans" : run k-means and use the hard labels
        "k-means++" : use the k-means++ seeding labels
        "random" : random responsibilities, normalized per sample
        "random_from_data" : pick n_components samples as one-hot
        responsibilities
    seed : int
        Seed to the random number generator.
    """

    cdef cuvsGMMParams* params

    def __cinit__(self):
        cuvsGMMParamsCreate(&self.params)

    def __dealloc__(self):
        check_cuvs(cuvsGMMParamsDestroy(self.params))

    def __init__(self, *,
                 n_components=None,
                 covariance_type=None,
                 tol=None,
                 reg_covar=None,
                 max_iter=None,
                 n_init=None,
                 init_method=None,
                 seed=None):
        if n_components is not None:
            self.params.n_components = n_components
        if covariance_type is not None:
            c_cov = COVARIANCE_TYPES[covariance_type]
            self.params.covariance_type = <cuvsGMMCovarianceType>c_cov
        if tol is not None:
            self.params.tol = tol
        if reg_covar is not None:
            self.params.reg_covar = reg_covar
        if max_iter is not None:
            self.params.max_iter = max_iter
        if n_init is not None:
            self.params.n_init = n_init
        if init_method is not None:
            c_init = INIT_METHOD_TYPES[init_method]
            self.params.init = <cuvsGMMInitMethod>c_init
        if seed is not None:
            self.params.seed = seed

    @property
    def n_components(self):
        return self.params.n_components

    @property
    def covariance_type(self):
        return COVARIANCE_NAMES[self.params.covariance_type]

    @property
    def tol(self):
        return self.params.tol

    @property
    def reg_covar(self):
        return self.params.reg_covar

    @property
    def max_iter(self):
        return self.params.max_iter

    @property
    def n_init(self):
        return self.params.n_init

    @property
    def init_method(self):
        return INIT_METHOD_NAMES[self.params.init]

    @property
    def seed(self):
        return self.params.seed


FitOutput = namedtuple(
    "FitOutput",
    "weights means covariances precisions_chol precisions labels "
    "lower_bound n_iter converged",
)


@auto_sync_resources
@auto_convert_output
def fit(GMMParams params, X, weights=None, means=None, covariances=None,
        warm_start=False, resources=None):
    """
    Fit a Gaussian mixture model with the EM algorithm

    Parameters
    ----------
    params : GMMParams
        Parameters of the EM solver.
    X : Input CUDA array interface compliant matrix shape (m, k)
    weights : Optional writable CUDA array interface vector,
              shape (n_components,). Holds the initial mixture weights when
              ``warm_start`` is True and receives the fitted weights.
    means : Optional writable CUDA array interface matrix,
            shape (n_components, k). Holds the initial means when
            ``warm_start`` is True and receives the fitted means.
    covariances : Optional writable CUDA array interface array whose shape
                  depends on ``params.covariance_type`` ("full":
                  (n_components, k, k), "tied": (k, k), "diag":
                  (n_components, k), "spherical": (n_components,)). Holds the
                  initial covariances when ``warm_start`` is True and receives
                  the fitted covariances.
    warm_start : bool
        Use the provided weights/means/covariances as the single
        initialization instead of running ``params.n_init`` restarts.
    {resources_docstring}

    Returns
    -------
    weights : raft.device_ndarray
        Fitted mixture weights, shape (n_components,)
    means : raft.device_ndarray
        Fitted component means, shape (n_components, k)
    covariances : raft.device_ndarray
        Fitted covariances (shape depends on covariance_type)
    precisions_chol : raft.device_ndarray
        Precision Cholesky factors (shape depends on covariance_type)
    precisions : raft.device_ndarray
        Precision matrices (shape depends on covariance_type)
    labels : raft.device_ndarray
        Hard component assignment per sample, shape (m,)
    lower_bound : float
        Per-sample average log-likelihood of the best fit
    n_iter : int
        Number of EM iterations of the best fit
    converged : bool
        Whether the best fit converged within ``params.tol``

    Examples
    --------

    >>> import cupy as cp
    >>>
    >>> from cuvs.cluster.gmm import fit, GMMParams
    >>>
    >>> n_samples = 5000
    >>> n_features = 50
    >>> n_components = 3
    >>>
    >>> X = cp.random.random_sample((n_samples, n_features),
    ...                             dtype=cp.float32)

    >>> params = GMMParams(n_components=n_components)
    >>> out = fit(params, X)
    >>> means = out.means
    """

    x_ai = wrap_array(X)
    _check_input_array(x_ai, [np.dtype('float32'), np.dtype('float64')])

    n_samples = x_ai.shape[0]
    n_features = x_ai.shape[1]
    n_components = params.n_components
    cov_shape = _covariance_shape(
        params.covariance_type, n_components, n_features)

    if weights is None:
        if warm_start:
            raise ValueError("warm_start requires initial weights")
        weights = device_ndarray.empty((n_components,), dtype=x_ai.dtype)
    if means is None:
        if warm_start:
            raise ValueError("warm_start requires initial means")
        means = device_ndarray.empty(
            (n_components, n_features), dtype=x_ai.dtype)
    if covariances is None:
        if warm_start:
            raise ValueError("warm_start requires initial covariances")
        covariances = device_ndarray.empty(cov_shape, dtype=x_ai.dtype)

    precisions_chol = device_ndarray.empty(cov_shape, dtype=x_ai.dtype)
    precisions = device_ndarray.empty(cov_shape, dtype=x_ai.dtype)
    labels = device_ndarray.empty((n_samples,), dtype='int32')

    weights_ai = wrap_array(weights)
    means_ai = wrap_array(means)
    covariances_ai = wrap_array(covariances)
    precisions_chol_ai = wrap_array(precisions_chol)
    precisions_ai = wrap_array(precisions)
    labels_ai = wrap_array(labels)

    _check_input_array(weights_ai, [x_ai.dtype], exp_rows=n_components)
    _check_input_array(means_ai, [x_ai.dtype], exp_rows=n_components,
                       exp_cols=n_features)
    _check_input_array(covariances_ai, [x_ai.dtype])
    _check_input_array(labels_ai, [np.dtype('int32')], exp_rows=n_samples)

    cdef cydlpack.DLManagedTensor* x_dlpack = cydlpack.dlpack_c(x_ai)
    cdef cydlpack.DLManagedTensor* weights_dlpack = \
        cydlpack.dlpack_c(weights_ai)
    cdef cydlpack.DLManagedTensor* means_dlpack = cydlpack.dlpack_c(means_ai)
    cdef cydlpack.DLManagedTensor* covariances_dlpack = \
        cydlpack.dlpack_c(covariances_ai)
    cdef cydlpack.DLManagedTensor* precisions_chol_dlpack = \
        cydlpack.dlpack_c(precisions_chol_ai)
    cdef cydlpack.DLManagedTensor* precisions_dlpack = \
        cydlpack.dlpack_c(precisions_ai)
    cdef cydlpack.DLManagedTensor* labels_dlpack = cydlpack.dlpack_c(labels_ai)

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef double lower_bound = 0
    cdef int n_iter = 0
    cdef bool converged = False
    cdef bool c_warm_start = warm_start

    with cuda_interruptible():
        check_cuvs(cuvsGMMFit(
            res,
            params.params,
            x_dlpack,
            weights_dlpack,
            means_dlpack,
            covariances_dlpack,
            precisions_chol_dlpack,
            precisions_dlpack,
            labels_dlpack,
            &lower_bound,
            &n_iter,
            &converged,
            c_warm_start))

    return FitOutput(weights, means, covariances, precisions_chol,
                     precisions, labels, lower_bound, n_iter,
                     True if converged else False)


@auto_sync_resources
@auto_convert_output
def predict(GMMParams params, X, weights, means, precisions_chol,
            labels=None, resources=None):
    """
    Hard component labels (argmax responsibility) for new data

    Parameters
    ----------
    params : GMMParams
        Parameters used to fit the GMM model.
    X : Input CUDA array interface compliant matrix shape (m, k)
    weights : Fitted mixture weights, shape (n_components,)
    means : Fitted component means, shape (n_components, k)
    precisions_chol : Fitted precision Cholesky factors (shape depends on
                      covariance_type)
    labels : Optional preallocated CUDA array interface vector shape (m,)
        to hold the output (int32)
    {resources_docstring}

    Returns
    -------
    labels : raft.device_ndarray
        Component assignment for each datapoint in X

    Examples
    --------

    >>> import cupy as cp
    >>>
    >>> from cuvs.cluster.gmm import fit, predict, GMMParams
    >>>
    >>> X = cp.random.random_sample((5000, 50), dtype=cp.float32)
    >>> params = GMMParams(n_components=3)
    >>> out = fit(params, X)
    >>>
    >>> labels = predict(params, X, out.weights, out.means,
    ...                  out.precisions_chol)
    """

    x_ai = wrap_array(X)
    _check_input_array(x_ai, [np.dtype('float32'), np.dtype('float64')])

    if labels is None:
        labels = device_ndarray.empty((x_ai.shape[0],), dtype='int32')

    labels_ai = wrap_array(labels)
    _check_input_array(
        labels_ai, [np.dtype('int32')], exp_rows=x_ai.shape[0])

    weights_ai = wrap_array(weights)
    means_ai = wrap_array(means)
    precisions_chol_ai = wrap_array(precisions_chol)
    _check_input_array(weights_ai, [x_ai.dtype])
    _check_input_array(means_ai, [x_ai.dtype], exp_rows=params.n_components,
                       exp_cols=x_ai.shape[1])
    _check_input_array(precisions_chol_ai, [x_ai.dtype])

    cdef cydlpack.DLManagedTensor* x_dlpack = cydlpack.dlpack_c(x_ai)
    cdef cydlpack.DLManagedTensor* weights_dlpack = \
        cydlpack.dlpack_c(weights_ai)
    cdef cydlpack.DLManagedTensor* means_dlpack = cydlpack.dlpack_c(means_ai)
    cdef cydlpack.DLManagedTensor* precisions_chol_dlpack = \
        cydlpack.dlpack_c(precisions_chol_ai)
    cdef cydlpack.DLManagedTensor* labels_dlpack = cydlpack.dlpack_c(labels_ai)

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        check_cuvs(cuvsGMMPredict(
            res,
            params.params,
            x_dlpack,
            weights_dlpack,
            means_dlpack,
            precisions_chol_dlpack,
            labels_dlpack))

    return labels


@auto_sync_resources
@auto_convert_output
def predict_proba(GMMParams params, X, weights, means, precisions_chol,
                  resp=None, resources=None):
    """
    Posterior responsibilities for new data

    Parameters
    ----------
    params : GMMParams
        Parameters used to fit the GMM model.
    X : Input CUDA array interface compliant matrix shape (m, k)
    weights : Fitted mixture weights, shape (n_components,)
    means : Fitted component means, shape (n_components, k)
    precisions_chol : Fitted precision Cholesky factors (shape depends on
                      covariance_type)
    resp : Optional preallocated CUDA array interface matrix shape
        (m, n_components) to hold the output
    {resources_docstring}

    Returns
    -------
    resp : raft.device_ndarray
        Posterior probability of each component for each sample

    Examples
    --------

    >>> import cupy as cp
    >>>
    >>> from cuvs.cluster.gmm import fit, predict_proba, GMMParams
    >>>
    >>> X = cp.random.random_sample((5000, 50), dtype=cp.float32)
    >>> params = GMMParams(n_components=3)
    >>> out = fit(params, X)
    >>>
    >>> resp = predict_proba(params, X, out.weights, out.means,
    ...                      out.precisions_chol)
    """

    x_ai = wrap_array(X)
    _check_input_array(x_ai, [np.dtype('float32'), np.dtype('float64')])

    if resp is None:
        resp = device_ndarray.empty(
            (x_ai.shape[0], params.n_components), dtype=x_ai.dtype)

    resp_ai = wrap_array(resp)
    _check_input_array(resp_ai, [x_ai.dtype], exp_rows=x_ai.shape[0],
                       exp_cols=params.n_components)

    weights_ai = wrap_array(weights)
    means_ai = wrap_array(means)
    precisions_chol_ai = wrap_array(precisions_chol)
    _check_input_array(weights_ai, [x_ai.dtype])
    _check_input_array(means_ai, [x_ai.dtype], exp_rows=params.n_components,
                       exp_cols=x_ai.shape[1])
    _check_input_array(precisions_chol_ai, [x_ai.dtype])

    cdef cydlpack.DLManagedTensor* x_dlpack = cydlpack.dlpack_c(x_ai)
    cdef cydlpack.DLManagedTensor* weights_dlpack = \
        cydlpack.dlpack_c(weights_ai)
    cdef cydlpack.DLManagedTensor* means_dlpack = cydlpack.dlpack_c(means_ai)
    cdef cydlpack.DLManagedTensor* precisions_chol_dlpack = \
        cydlpack.dlpack_c(precisions_chol_ai)
    cdef cydlpack.DLManagedTensor* resp_dlpack = cydlpack.dlpack_c(resp_ai)

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        check_cuvs(cuvsGMMPredictProba(
            res,
            params.params,
            x_dlpack,
            weights_dlpack,
            means_dlpack,
            precisions_chol_dlpack,
            resp_dlpack))

    return resp


@auto_sync_resources
@auto_convert_output
def score_samples(GMMParams params, X, weights, means, precisions_chol,
                  log_prob=None, resources=None):
    """
    Per-sample log-likelihood log p(x_i) for new data

    Parameters
    ----------
    params : GMMParams
        Parameters used to fit the GMM model.
    X : Input CUDA array interface compliant matrix shape (m, k)
    weights : Fitted mixture weights, shape (n_components,)
    means : Fitted component means, shape (n_components, k)
    precisions_chol : Fitted precision Cholesky factors (shape depends on
                      covariance_type)
    log_prob : Optional preallocated CUDA array interface vector shape (m,)
        to hold the output
    {resources_docstring}

    Returns
    -------
    log_prob : raft.device_ndarray
        Log-likelihood of each sample under the model

    Examples
    --------

    >>> import cupy as cp
    >>>
    >>> from cuvs.cluster.gmm import fit, score_samples, GMMParams
    >>>
    >>> X = cp.random.random_sample((5000, 50), dtype=cp.float32)
    >>> params = GMMParams(n_components=3)
    >>> out = fit(params, X)
    >>>
    >>> log_prob = score_samples(params, X, out.weights, out.means,
    ...                          out.precisions_chol)
    """

    x_ai = wrap_array(X)
    _check_input_array(x_ai, [np.dtype('float32'), np.dtype('float64')])

    if log_prob is None:
        log_prob = device_ndarray.empty((x_ai.shape[0],), dtype=x_ai.dtype)

    log_prob_ai = wrap_array(log_prob)
    _check_input_array(log_prob_ai, [x_ai.dtype], exp_rows=x_ai.shape[0])

    weights_ai = wrap_array(weights)
    means_ai = wrap_array(means)
    precisions_chol_ai = wrap_array(precisions_chol)
    _check_input_array(weights_ai, [x_ai.dtype])
    _check_input_array(means_ai, [x_ai.dtype], exp_rows=params.n_components,
                       exp_cols=x_ai.shape[1])
    _check_input_array(precisions_chol_ai, [x_ai.dtype])

    cdef cydlpack.DLManagedTensor* x_dlpack = cydlpack.dlpack_c(x_ai)
    cdef cydlpack.DLManagedTensor* weights_dlpack = \
        cydlpack.dlpack_c(weights_ai)
    cdef cydlpack.DLManagedTensor* means_dlpack = cydlpack.dlpack_c(means_ai)
    cdef cydlpack.DLManagedTensor* precisions_chol_dlpack = \
        cydlpack.dlpack_c(precisions_chol_ai)
    cdef cydlpack.DLManagedTensor* log_prob_dlpack = \
        cydlpack.dlpack_c(log_prob_ai)

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        check_cuvs(cuvsGMMScoreSamples(
            res,
            params.params,
            x_dlpack,
            weights_dlpack,
            means_dlpack,
            precisions_chol_dlpack,
            log_prob_dlpack))

    return log_prob
