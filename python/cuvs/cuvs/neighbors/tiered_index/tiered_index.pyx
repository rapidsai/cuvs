#
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
# cython: language_level=3

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
from cuvs.neighbors import cagra, ivf_flat, ivf_pq
from cuvs.neighbors.common import _check_input_array
from cuvs.neighbors.filters import no_filter

from libc.stdint cimport (
    int8_t,
    int64_t,
    uint8_t,
    uint32_t,
    uint64_t,
    uintptr_t,
)

from cuvs.common.exceptions import check_cuvs

ALGO_TYPES = {
    "cagra": CUVS_TIERED_INDEX_ALGO_CAGRA,
    "ivf_flat": CUVS_TIERED_INDEX_ALGO_IVF_FLAT,
    "ivf_pq": CUVS_TIERED_INDEX_ALGO_IVF_PQ
}
ALGO_NAMES = {v: k for k, v in ALGO_TYPES.items()}


cdef class IndexParams:
    """
    Parameters to build index for Tiered Index nearest neighbor search

    Parameters
    ----------
    metric : str, default = "sqeuclidean"
        String denoting the metric type.
        Valid values for metric: ["sqeuclidean", "inner_product",
        "euclidean", "cosine"], where
            - sqeuclidean is the euclidean distance without the square root
              operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2,
            - euclidean is the euclidean distance
            - inner product distance is defined as
              distance(a, b) = \\sum_i a_i * b_i.
            - cosine distance is defined as
              distance(a, b) = 1 - \\sum_i a_i * b_i / ( ||a||_2 * ||b||_2).
    algo : str, default = "cagra"
        The algorithm to use for the ANN portion of the tiered index
    upstream_params : object, optional
        The IndexParams for the upstream ANN object to use (ie the Cagra
        IndexParams for cagra etc)
    min_ann_rows : int
        The minimum number of rows necessary to create an ann index
    create_ann_index_on_extend : bool
        Whether or not to create a new ann index on extend, if the number
        of rows in the incremental (bfknn) portion is above min_ann_rows
    """

    cdef cuvsTieredIndexParams* params
    cdef object _upstream_params

    def __cinit__(self):
        cuvsTieredIndexParamsCreate(&self.params)

    def __dealloc__(self):
        check_cuvs(cuvsTieredIndexParamsDestroy(self.params))

    def __init__(self, *,
                 metric="sqeuclidean",
                 algo="cagra",
                 upstream_params=None,
                 min_ann_rows=None,
                 create_ann_index_on_extend=None,):
        self.params.metric = <cuvsDistanceType>DISTANCE_TYPES[metric]
        self.params.algo = <cuvsTieredIndexANNAlgo>ALGO_TYPES[algo]
        if min_ann_rows is not None:
            self.params.min_ann_rows = min_ann_rows
        if create_ann_index_on_extend is not None:
            self.params.create_ann_index_on_extend = create_ann_index_on_extend

        self._upstream_params = upstream_params
        if upstream_params is not None:
            if algo == "cagra":
                if not isinstance(upstream_params, cagra.IndexParams):
                    raise TypeError("Expected cagra.IndexParams, got "
                                    f"{upstream_params.__class__} ")
                h = upstream_params.get_handle()
                self.params.cagra_params = <cuvsCagraIndexParams_t><size_t> h
            elif algo == "ivf_flat":
                if not isinstance(upstream_params, ivf_flat.IndexParams):
                    raise TypeError("Expected ivf_flat.IndexParams, got "
                                    f"{upstream_params.__class__} ")
                h = upstream_params.get_handle()
                self.params.ivf_flat_params = \
                    <cuvsIvfFlatIndexParams_t><size_t> h
            elif algo == "ivf_pq":
                if not isinstance(upstream_params, ivf_pq.IndexParams):
                    raise TypeError("Expected ivf_pq.IndexParams, got "
                                    f"{upstream_params.__class__} ")
                h = upstream_params.get_handle()
                self.params.ivf_pq_params = <cuvsIvfPqIndexParams_t><size_t> h
            else:
                raise ValueError(f"Unknown algorithm '{algo}")

    @property
    def metric(self):
        return DISTANCE_NAMES[self.params.metric]

    @property
    def algo(self):
        return ALGO_NAMES[self.params.algo]

    @property
    def min_ann_rows(self):
        return self.params.min_ann_rows

    @property
    def create_ann_index_on_extend(self):
        return self.params.create_ann_index_on_extend

    @property
    def upstream_params(self):
        return self._upstream_params

cdef class Index:
    """
    Tiered Index object.
    """

    cdef cuvsTieredIndex_t index
    cdef bool trained

    def __cinit__(self):
        self.trained = False
        check_cuvs(cuvsTieredIndexCreate(&self.index))

    def __dealloc__(self):
        check_cuvs(cuvsTieredIndexDestroy(self.index))

    @property
    def trained(self):
        return self.trained


@auto_sync_resources
def build(IndexParams index_params, dataset, resources=None):
    """
    Build the Tiered index from the dataset for efficient search.

    Parameters
    ----------
    index_params : :py:class:`cuvs.neighbors.tiered_index.IndexParams`
    dataset : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float32]
    {resources_docstring}

    Returns
    -------
    index: py:class:`cuvs.neighbors.tiered_index.Index`

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.neighbors import cagra, tiered_index
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> k = 10
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> build_params = tiered_index.IndexParams(metric="sqeuclidean",
    ...                                         algo="cagra")
    >>> index = tiered_index.build(build_params, dataset)
    >>> distances, neighbors = tiered_index.search(cagra.SearchParams(),
    ...                                            index, dataset, k)
    >>> distances = cp.asarray(distances)
    >>> neighbors = cp.asarray(neighbors)
    """

    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype('float32')])

    cdef Index idx = Index()
    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cuvsTieredIndexParams* params = index_params.params

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        check_cuvs(cuvsTieredIndexBuild(
            res,
            params,
            dataset_dlpack,
            idx.index
        ))
        idx.trained = True

    return idx


@auto_sync_resources
@auto_convert_output
def search(search_params,
           Index index,
           queries,
           k,
           neighbors=None,
           distances=None,
           resources=None,
           filter=None):
    """
    Find the k nearest neighbors for each query.

    Parameters
    ----------
    search_params : SearchParams for the upstream ANN index
    index : py:class:`cuvs.neighbors.tiered_index.Index`
        Trained Tiered index.
    queries : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float32]
    k : int
        The number of neighbors.
    neighbors : Optional CUDA array interface compliant matrix shape
                (n_queries, k), dtype int64_t. If supplied, neighbor
                indices will be written here in-place. (default None)
    distances : Optional CUDA array interface compliant matrix shape
                (n_queries, k) If supplied, the distances to the
                neighbors will be written here in-place. (default None)
    filter:     Optional cuvs.neighbors.cuvsFilter can be used to filter
                neighbors based on a given bitset. (default None)
    {resources_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.neighbors import cagra, tiered_index
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build the index
    >>> index = tiered_index.build(tiered_index.IndexParams(algo="cagra"),
    ...                            dataset)
    >>>
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 10
    >>> search_params = cagra.SearchParams()
    >>>
    >>> distances, neighbors = tiered_index.search(search_params, index,
    ...                                            queries, k)
    """
    if not index.trained:
        raise ValueError("Index needs to be built before calling search.")

    queries_cai = wrap_array(queries)
    _check_input_array(queries_cai, [np.dtype('float32')])

    cdef uint32_t n_queries = queries_cai.shape[0]

    if neighbors is None:
        neighbors = device_ndarray.empty((n_queries, k), dtype='int64')

    neighbors_cai = wrap_array(neighbors)
    _check_input_array(neighbors_cai, [np.dtype('int64')],
                       exp_rows=n_queries, exp_cols=k)

    if distances is None:
        distances = device_ndarray.empty((n_queries, k), dtype='float32')

    distances_cai = wrap_array(distances)
    _check_input_array(distances_cai, [np.dtype('float32')],
                       exp_rows=n_queries, exp_cols=k)

    if filter is None:
        filter = no_filter()

    cdef void* params = <void*><size_t>search_params.get_handle()

    cdef cuvsError_t search_status
    cdef cydlpack.DLManagedTensor* queries_dlpack = \
        cydlpack.dlpack_c(queries_cai)
    cdef cydlpack.DLManagedTensor* neighbors_dlpack = \
        cydlpack.dlpack_c(neighbors_cai)
    cdef cydlpack.DLManagedTensor* distances_dlpack = \
        cydlpack.dlpack_c(distances_cai)
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        check_cuvs(cuvsTieredIndexSearch(
            res,
            params,
            index.index,
            queries_dlpack,
            neighbors_dlpack,
            distances_dlpack,
            filter.prefilter
        ))

    return (distances, neighbors)


@auto_sync_resources
def extend(Index index, new_vectors, resources=None):
    """
    Extend an existing index with new vectors.

    The input array can be either CUDA array interface compliant matrix or
    array interface compliant matrix in host memory.


    Parameters
    ----------
    index : tiered_index.Index
        Trained tiered_index object.
    new_vectors : array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float32]
    {resources_docstring}

    Returns
    -------
    index: py:class:`cuvs.neighbors.tiered_index.Index`

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.neighbors import tiered_index
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> index = tiered_index.build(tiered_index.IndexParams(), dataset)
    >>> n_rows = 100
    >>> more_data = cp.random.random_sample((n_rows, n_features),
    ...                                     dtype=cp.float32)
    >>> index = tiered_index.extend(index, more_data)
    """

    new_vectors_ai = wrap_array(new_vectors)
    _check_input_array(new_vectors_ai, [np.dtype('float32')])

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef cydlpack.DLManagedTensor* new_vectors_dlpack = \
        cydlpack.dlpack_c(new_vectors_ai)

    with cuda_interruptible():
        check_cuvs(cuvsTieredIndexExtend(
            res,
            new_vectors_dlpack,
            index.index
        ))

    return index
