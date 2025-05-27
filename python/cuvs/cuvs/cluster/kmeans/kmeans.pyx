#
# Copyright (c) 2024, NVIDIA CORPORATION.
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
# cython: language_level=3

from collections import namedtuple

import numpy as np

cimport cuvs.common.cydlpack

from cuvs.common.resources import auto_sync_resources

from cython.operator cimport dereference as deref
from libcpp cimport bool, cast
from libcpp.string cimport string

from cuvs.common cimport cydlpack
from cuvs.distance_type cimport cuvsDistanceType

from pylibraft.common import auto_convert_output, cai_wrapper, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible

from cuvs.distance import DISTANCE_NAMES, DISTANCE_TYPES
from cuvs.neighbors.common import _check_input_array

from libc.stdint cimport (
    int8_t,
    int64_t,
    uint8_t,
    uint32_t,
    uint64_t,
    uintptr_t,
)

from cuvs.common.exceptions import check_cuvs

INIT_METHOD_TYPES = {
    "KMeansPlusPlus" : cuvsKMeansInitMethod.KMeansPlusPlus,
    "Random" : cuvsKMeansInitMethod.Random,
    "Array" : cuvsKMeansInitMethod.Array}

INIT_METHOD_NAMES = {v: k for k, v in INIT_METHOD_TYPES.items()}

cdef class KMeansParams:
    """
    Hyper-parameters for the kmeans algorithm

    Parameters
    ----------
    metric : str
        String denoting the metric type.
    n_clusters : int
        The number of clusters to form as well as the number of centroids
        to generate
    init_method : str
        Method for initializing clusters. One of:
        "KMeansPlusPlus" : Use scalable k-means++ algorithm to select initial
        cluster centers
        "Random" : Choose 'n_clusters' observations at random from the input
        data
        "Array" : Use centroids as initial cluster centers
    max_iter : int
        Maximum number of iterations of the k-means algorithm for a single run
    tol : float
        Relative tolerance with regards to inertia to declare convergence.
    n_init : int
        Number of instance k-means algorithm will be run with different seeds
    oversampling_factor : double
        Oversampling factor for use in the k-means|| algorithm
    hierarchical : bool
        Whether to use hierarchical (balanced) kmeans or not
    hierarchical_n_iters : int
        For hierarchical k-means , defines the number of training iterations
    """

    cdef cuvsKMeansParams* params

    def __cinit__(self):
        cuvsKMeansParamsCreate(&self.params)

    def __dealloc__(self):
        check_cuvs(cuvsKMeansParamsDestroy(self.params))

    def __init__(self, *,
                 metric=None,
                 n_clusters=None,
                 init_method=None,
                 max_iter=None,
                 tol=None,
                 n_init=None,
                 oversampling_factor=None,
                 hierarchical=None,
                 hierarchical_n_iters=None):
        if metric is not None:
            self.params.metric = <cuvsDistanceType>DISTANCE_TYPES[metric]
        if n_clusters is not None:
            self.params.n_clusters = n_clusters
        if init_method is not None:
            c_method = INIT_METHOD_TYPES[init_method]
            self.params.init = <cuvsKMeansInitMethod>c_method
        if max_iter is not None:
            self.params.max_iter = max_iter
        if tol is not None:
            self.params.tol = tol
        if n_init is not None:
            self.params.n_init = n_init
        if oversampling_factor is not None:
            self.params.oversampling_factor = oversampling_factor
        if hierarchical is not None:
            self.params.hierarchical = hierarchical
        if hierarchical_n_iters is not None:
            if not self.params.hierarchical:
                raise ValueError("Setting hierarchical_n_iters requires"
                                 " `hierarchical` to be also set to True")
            self.params.hierarchical_n_iters = hierarchical_n_iters

    @property
    def metric(self):
        return DISTANCE_NAMES[self.params.metric]

    @property
    def n_clusters(self):
        return self.params.n_clusters

    @property
    def init_method(self):
        return INIT_METHOD_NAMES[self.params.init]

    @property
    def max_iter(self):
        return self.params.max_iter

    @property
    def tol(self):
        return self.params.tol

    @property
    def n_init(self):
        return self.params.n_init

    @property
    def oversampling_factor(self):
        return self.params.oversampling_factor

    @property
    def hierarchical(self):
        return self.params.hierarchical

    @property
    def hierarchical_n_iters(self):
        return self.params.hierarchical_n_iters


FitOutput = namedtuple("FitOutput", "centroids inertia n_iter")


@auto_sync_resources
@auto_convert_output
def fit(
    KMeansParams params, X, centroids=None, sample_weights=None, resources=None
):
    """
    Find clusters with the k-means algorithm

    Parameters
    ----------

    params : KMeansParams
        Parameters to use to fit KMeans model
    X : Input CUDA array interface compliant matrix shape (m, k)
    centroids : Optional writable CUDA array interface compliant matrix
                shape (n_clusters, k)
    sample_weights : Optional input CUDA array interface compliant matrix shape
                     (n_clusters, 1) default: None
    {resources_docstring}

    Returns
    -------
    centroids : raft.device_ndarray
        The computed centroids for each cluster
    inertia : float
       Sum of squared distances of samples to their closest cluster center
    n_iter : int
        The number of iterations used to fit the model

    Examples
    --------

    >>> import cupy as cp
    >>>
    >>> from cuvs.cluster.kmeans import fit, KMeansParams
    >>>
    >>> n_samples = 5000
    >>> n_features = 50
    >>> n_clusters = 3
    >>>
    >>> X = cp.random.random_sample((n_samples, n_features),
    ...                             dtype=cp.float32)

    >>> params = KMeansParams(n_clusters=n_clusters)
    >>> centroids, inertia, n_iter = fit(params, X)
    """

    x_ai = wrap_array(X)
    _check_input_array(x_ai, [np.dtype('float32'), np.dtype('float64')])

    cdef cydlpack.DLManagedTensor* x_dlpack = cydlpack.dlpack_c(x_ai)
    cdef cydlpack.DLManagedTensor* sample_weight_dlpack = NULL

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef double inertia = 0
    cdef int n_iter = 0

    if centroids is None:
        centroids = device_ndarray.empty((params.n_clusters, x_ai.shape[1]),
                                         dtype=x_ai.dtype)

    centroids_ai = wrap_array(centroids)
    cdef cydlpack.DLManagedTensor * centroids_dlpack = \
        cydlpack.dlpack_c(centroids_ai)

    if sample_weights is not None:
        sample_weight_dlpack = cydlpack.dlpack_c(wrap_array(sample_weights))

    with cuda_interruptible():
        check_cuvs(cuvsKMeansFit(
            res,
            params.params,
            x_dlpack,
            sample_weight_dlpack,
            centroids_dlpack,
            &inertia,
            &n_iter))

    return FitOutput(centroids, inertia, n_iter)


PredictOutput = namedtuple("PredictOutput", "labels inertia")


@auto_sync_resources
@auto_convert_output
def predict(
    KMeansParams params, X, centroids, sample_weights=None, labels=None,
    normalize_weight=True, resources=None
):
    """
    Predict clusters with the k-means algorithm

    Parameters
    ----------

    params : KMeansParams
        Parameters to used in fitting KMeans model
    X : Input CUDA array interface compliant matrix shape (m, k)
    centroids : CUDA array interface compliant matrix, calculated by fit
                shape (n_clusters, k)
    sample_weights : Optional input CUDA array interface compliant matrix shape
                     (n_clusters, 1) default: None
    labels : Optional preallocated CUDA array interface matrix shape (m, 1)
        to hold the output
    normalize_weight: bool
        True if the weights should be normalized
    {resources_docstring}

    Returns
    -------
    labels : raft.device_ndarray
        The label for each datapoint in X
    inertia : float
       Sum of squared distances of samples to their closest cluster center

    Examples
    --------

    >>> import cupy as cp
    >>>
    >>> from cuvs.cluster.kmeans import fit, predict, KMeansParams
    >>>
    >>> n_samples = 5000
    >>> n_features = 50
    >>> n_clusters = 3
    >>>
    >>> X = cp.random.random_sample((n_samples, n_features),
    ...                             dtype=cp.float32)

    >>> params = KMeansParams(n_clusters=n_clusters)
    >>> centroids, inertia, n_iter = fit(params, X)
    >>>
    >>> labels, inertia = predict(params, X, centroids)
    """

    x_ai = wrap_array(X)
    _check_input_array(x_ai, [np.dtype('float32'), np.dtype('float64')])
    cdef cydlpack.DLManagedTensor* x_dlpack = cydlpack.dlpack_c(x_ai)

    cdef cydlpack.DLManagedTensor* sample_weight_dlpack = NULL
    if sample_weights is not None:
        sample_weight_dlpack = cydlpack.dlpack_c(wrap_array(sample_weights))

    if labels is None:
        labels = device_ndarray.empty((x_ai.shape[0]), dtype='int32')

    labels_ai = wrap_array(labels)
    _check_input_array(labels_ai, [np.dtype('int32')])
    cdef cydlpack.DLManagedTensor * labels_dlpack = \
        cydlpack.dlpack_c(labels_ai)

    centroids_ai = wrap_array(centroids)
    _check_input_array(centroids_ai, [np.dtype('float32'),
                                      np.dtype('float64')])
    cdef cydlpack.DLManagedTensor * centroids_dlpack = \
        cydlpack.dlpack_c(centroids_ai)

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef double inertia = 0

    with cuda_interruptible():
        check_cuvs(cuvsKMeansPredict(
            res,
            params.params,
            x_dlpack,
            sample_weight_dlpack,
            centroids_dlpack,
            labels_dlpack,
            normalize_weight,
            &inertia))

    return PredictOutput(labels, inertia)


@auto_sync_resources
@auto_convert_output
def cluster_cost(X, centroids, resources=None):
    """
    Compute cluster cost given an input matrix and existing centroids

    Parameters
    ----------
    X : Input CUDA array interface compliant matrix shape (m, k)
    centroids : Input CUDA array interface compliant matrix shape
                    (n_clusters, k)
    {resources_docstring}

    Returns
    -------
    inertia : float
        The cluster cost between the input matrix and existing centroids

    Examples
    --------

    >>> import cupy as cp
    >>>
    >>> from cuvs.cluster.kmeans import cluster_cost
    >>>
    >>> n_samples = 5000
    >>> n_features = 50
    >>> n_clusters = 3
    >>>
    >>> X = cp.random.random_sample((n_samples, n_features),
    ...                             dtype=cp.float32)

    >>> centroids = cp.random.random_sample((n_clusters, n_features),
    ...                                      dtype=cp.float32)

    >>> inertia = cluster_cost(X, centroids)
    """

    x_ai = wrap_array(X)
    _check_input_array(x_ai, [np.dtype('float32'), np.dtype('float64')])
    cdef cydlpack.DLManagedTensor* x_dlpack = cydlpack.dlpack_c(x_ai)

    centroids_ai = wrap_array(centroids)
    _check_input_array(centroids_ai, [np.dtype('float32'),
                                      np.dtype('float64')])
    cdef cydlpack.DLManagedTensor* centroids_dlpack = \
        cydlpack.dlpack_c(centroids_ai)

    cdef double inertia = 0
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        check_cuvs(cuvsKMeansClusterCost(
            res,
            x_dlpack,
            centroids_dlpack,
            &inertia))

    return inertia
