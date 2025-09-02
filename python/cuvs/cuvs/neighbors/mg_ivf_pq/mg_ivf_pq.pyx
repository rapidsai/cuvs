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

from libc.stdint cimport uint32_t
from libcpp.string cimport string

from pylibraft.common import auto_convert_output
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible

from cuvs.common.exceptions import check_cuvs
from cuvs.common.mg_resources import auto_sync_multi_gpu_resources
from cuvs.neighbors.common import _check_input_array, _check_memory_location

from cuvs.common cimport cydlpack
from cuvs.common.c_api cimport cuvsResources_t
from cuvs.neighbors.ivf_pq.ivf_pq cimport (
    IndexParams,
    SearchParams,
    cuvsIvfPqIndexParams_t,
    cuvsIvfPqIndexParamsDestroy,
    cuvsIvfPqSearchParams_t,
    cuvsIvfPqSearchParamsDestroy,
)
from cuvs.neighbors.mg_ivf_pq.mg_ivf_pq cimport (
    cuvsMultiGpuDistributionMode,
    cuvsMultiGpuIvfPqBuild,
    cuvsMultiGpuIvfPqDeserialize,
    cuvsMultiGpuIvfPqDistribute,
    cuvsMultiGpuIvfPqExtend,
    cuvsMultiGpuIvfPqIndex,
    cuvsMultiGpuIvfPqIndex_t,
    cuvsMultiGpuIvfPqIndexCreate,
    cuvsMultiGpuIvfPqIndexDestroy,
    cuvsMultiGpuIvfPqIndexParams,
    cuvsMultiGpuIvfPqIndexParams_t,
    cuvsMultiGpuIvfPqIndexParamsCreate,
    cuvsMultiGpuIvfPqIndexParamsDestroy,
    cuvsMultiGpuIvfPqSearch,
    cuvsMultiGpuIvfPqSearchParams,
    cuvsMultiGpuIvfPqSearchParams_t,
    cuvsMultiGpuIvfPqSearchParamsCreate,
    cuvsMultiGpuIvfPqSearchParamsDestroy,
    cuvsMultiGpuIvfPqSerialize,
    cuvsMultiGpuReplicatedSearchMode,
    cuvsMultiGpuShardedMergeMode,
)


cdef class MultiGpuIndexParams(IndexParams):
    """
    Parameters to build multi-GPU IVF-PQ index for efficient search.
    Extends single-GPU IndexParams with multi-GPU specific parameters.

    Parameters
    ----------
    distribution_mode : str, default = "sharded"
        Distribution mode for multi-GPU setup.
        Valid values: ["replicated", "sharded"]
    **kwargs : Additional parameters passed to single-GPU IndexParams
    """

    def __cinit__(self):
        # Base class __cinit__ has already created self.params
        # We need to destroy it and use our embedded params instead
        if self.params is not NULL:
            check_cuvs(cuvsIvfPqIndexParamsDestroy(self.params))

        # Create multi-GPU params which includes embedded base params
        check_cuvs(cuvsMultiGpuIvfPqIndexParamsCreate(&self.mg_params))
        # Replace base pointer with embedded base params
        self.params = self.mg_params.base_params

    def __dealloc__(self):
        # Only destroy the mg_params, which will handle base_params cleanup
        check_cuvs(cuvsMultiGpuIvfPqIndexParamsDestroy(self.mg_params))
        self.mg_params = NULL
        self.params = <cuvsIvfPqIndexParams_t>NULL

    def __init__(self, *, distribution_mode="sharded", **kwargs):
        super().__init__(**kwargs)
        if distribution_mode == "replicated":
            self.mg_params.mode = CUVS_NEIGHBORS_MG_REPLICATED
        elif distribution_mode == "sharded":
            self.mg_params.mode = CUVS_NEIGHBORS_MG_SHARDED
        else:
            raise ValueError(
                "distribution_mode must be 'replicated' or 'sharded'")

    def get_handle(self):
        return <size_t> self.mg_params

    @property
    def distribution_mode(self):
        return ("replicated" if self.mg_params.mode ==
                CUVS_NEIGHBORS_MG_REPLICATED else "sharded")


cdef class MultiGpuIndex:
    """
    Multi-GPU IVF-PQ index object. Stores the trained multi-GPU IVF-PQ
    index state which can be used to perform nearest neighbors searches
    across multiple GPUs.
    """

    def __cinit__(self):
        # Initialize multi-GPU index
        check_cuvs(cuvsMultiGpuIvfPqIndexCreate(&self.mg_index))
        # Initialize multi-GPU trained state
        self.mg_trained = False

    def __dealloc__(self):
        check_cuvs(cuvsMultiGpuIvfPqIndexDestroy(self.mg_index))

    def __repr__(self):
        return "Index(type=MultiGpuIvfPq)"

    @property
    def trained(self):
        return self.mg_trained


@auto_sync_multi_gpu_resources
def build(MultiGpuIndexParams index_params, dataset, resources=None):
    """
    Build the multi-GPU IVF-PQ index from the dataset for efficient search.

    Parameters
    ----------
    index_params : :py:class:`cuvs.neighbors.mg_ivf_pq.MultiGpuIndexParams`
    dataset : Array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float32, float16, int8, uint8]
        **IMPORTANT**: For multi-GPU IVF-PQ, the dataset MUST be
        in host memory (CPU).
        If using CuPy/device arrays, transfer to host with array.get()
        or cp.asnumpy(array).
    {resources_docstring}

    Returns
    -------
    index: py:class:`cuvs.neighbors.mg_ivf_pq.MultiGpuIndex`

    Examples
    --------

    >>> import numpy as np
    >>> from cuvs.neighbors import mg_ivf_pq
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> k = 10
    >>> # For multi-GPU IVF-PQ, use host (NumPy) arrays
    >>> dataset = np.random.random_sample((n_samples, n_features),
    ...                                   dtype=np.float32)
    >>> build_params = mg_ivf_pq.MultiGpuIndexParams(metric="sqeuclidean")
    >>> index = mg_ivf_pq.build(build_params, dataset)
    >>> distances, neighbors = mg_ivf_pq.search(
    ...     mg_ivf_pq.MultiGpuSearchParams(),
    ...     index, dataset, k)
    >>> # Results are already in host memory (NumPy arrays)
    """

    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype('float32'), np.dtype('float16'),
                                    np.dtype('byte'), np.dtype('ubyte')])

    # Multi-GPU IVF-PQ requires dataset in host memory
    _check_memory_location(dataset, expected_host=True, name="dataset")

    cdef MultiGpuIndex idx = MultiGpuIndex()
    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cuvsMultiGpuIvfPqIndexParams_t params = index_params.mg_params

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    # Build the multi-GPU index
    with cuda_interruptible():
        check_cuvs(cuvsMultiGpuIvfPqBuild(res, params,
                                          dataset_dlpack,
                                          idx.mg_index))
        idx.mg_trained = True

    return idx


cdef class MultiGpuSearchParams(SearchParams):
    """
    Parameters to search multi-GPU IVF-PQ index.
    """

    def __cinit__(self):
        # Base class __cinit__ has already created self.params
        # We need to destroy it and use our embedded params instead
        if self.params is not NULL:
            check_cuvs(cuvsIvfPqSearchParamsDestroy(self.params))

        # Create multi-GPU search params which includes embedded base params
        check_cuvs(cuvsMultiGpuIvfPqSearchParamsCreate(&self.mg_params))
        # Replace base pointer with embedded base params
        self.params = self.mg_params.base_params

    def __dealloc__(self):
        # Only destroy the mg_params, which will handle base_params cleanup
        check_cuvs(cuvsMultiGpuIvfPqSearchParamsDestroy(self.mg_params))
        self.mg_params = NULL
        self.params = <cuvsIvfPqSearchParams_t>NULL

    def __init__(self, *, n_probes=20, search_mode="load_balancer",
                 merge_mode="merge_on_root_rank", n_rows_per_batch=1000,
                 **kwargs):
        super().__init__(n_probes=n_probes, **kwargs)
        if search_mode == "load_balancer":
            self.mg_params.search_mode = CUVS_NEIGHBORS_MG_LOAD_BALANCER
        elif search_mode == "round_robin":
            self.mg_params.search_mode = CUVS_NEIGHBORS_MG_ROUND_ROBIN
        else:
            raise ValueError(
                "search_mode must be 'load_balancer' or 'round_robin'")
        if merge_mode == "merge_on_root_rank":
            self.mg_params.merge_mode = CUVS_NEIGHBORS_MG_MERGE_ON_ROOT_RANK
        elif merge_mode == "tree_merge":
            self.mg_params.merge_mode = CUVS_NEIGHBORS_MG_TREE_MERGE
        else:
            raise ValueError(
                "merge_mode must be 'merge_on_root_rank' or 'tree_merge'"
            )
        self.mg_params.n_rows_per_batch = n_rows_per_batch

    def get_handle(self):
        return <size_t> self.mg_params


@auto_sync_multi_gpu_resources
@auto_convert_output
def search(MultiGpuSearchParams search_params, MultiGpuIndex index, queries,
           k, neighbors=None, distances=None, resources=None):
    """
    Search the multi-GPU IVF-PQ index for the k-nearest neighbors
    of each query.

    Parameters
    ----------
    search_params : :py:class:`cuvs.neighbors.mg_ivf_pq.MultiGpuSearchParams`
    index : :py:class:`cuvs.neighbors.mg_ivf_pq.MultiGpuIndex`
    queries : Array interface compliant matrix shape (n_queries, dim)
        Supported dtype [float32, float16, int8, uint8]
        **IMPORTANT**: For multi-GPU IVF-PQ, queries MUST be
        in host memory (CPU).
        If using CuPy/device arrays, transfer to host with array.get()
        or cp.asnumpy(array).
    k : int
        The number of neighbors to search for each query.
    neighbors : Array interface compliant matrix shape (n_queries, k), optional
        If provided, this array will be filled with the indices of
        the k-nearest neighbors.
        If not provided, a new host array will be allocated.
        **IMPORTANT**: Must be in host memory (CPU) for multi-GPU IVF-PQ.
    distances : Array interface compliant matrix shape (n_queries, k), optional
        If provided, this array will be filled with the distances to
        the k-nearest neighbors.
        If not provided, a new host array will be allocated.
        **IMPORTANT**: Must be in host memory (CPU) for multi-GPU IVF-PQ.
    {resources_docstring}

    Returns
    -------
    distances : numpy.ndarray
        The distances to the k-nearest neighbors for each query
        (in host memory).
    neighbors : numpy.ndarray
        The indices of the k-nearest neighbors for each query
        (in host memory).

    Examples
    --------

    >>> import numpy as np
    >>> from cuvs.neighbors import mg_ivf_pq
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> k = 10
    >>> # For multi-GPU IVF-PQ, use host (NumPy) arrays
    >>> dataset = np.random.random_sample((n_samples, n_features),
    ...                                   dtype=np.float32)
    >>> queries = np.random.random_sample((n_queries, n_features),
    ...                                   dtype=np.float32)
    >>> build_params = mg_ivf_pq.MultiGpuIndexParams(metric="sqeuclidean")
    >>> index = mg_ivf_pq.build(build_params, dataset)
    >>> distances, neighbors = \
    ... mg_ivf_pq.search(mg_ivf_pq.MultiGpuSearchParams(),
    ...                  index, queries, k)
    >>> # Results are already in host memory (NumPy arrays)
    """

    if not index.trained:
        raise ValueError("Index needs to be built before searching")

    queries_ai = wrap_array(queries)
    _check_input_array(queries_ai, [np.dtype('float32'), np.dtype('float16'),
                                    np.dtype('byte'), np.dtype('ubyte')])

    # Multi-GPU IVF-PQ requires queries in host memory
    _check_memory_location(queries, expected_host=True, name="queries")

    # Get resources
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    # Prepare output arrays
    cdef uint32_t n_queries = queries.shape[0]
    if neighbors is None:
        # For multi-GPU, create host arrays instead of device arrays
        neighbors = np.empty((n_queries, k), dtype='int64')
    if distances is None:
        # For multi-GPU, create host arrays instead of device arrays
        distances = np.empty((n_queries, k), dtype='float32')

    neighbors_ai = wrap_array(neighbors)
    _check_input_array(neighbors_ai, [np.dtype('int64')],
                       exp_rows=n_queries, exp_cols=k)
    distances_ai = wrap_array(distances)
    _check_input_array(distances_ai, [np.dtype('float32')],
                       exp_rows=n_queries, exp_cols=k)

    # Multi-GPU IVF-PQ requires output arrays in host memory
    _check_memory_location(neighbors, expected_host=True, name="neighbors")
    _check_memory_location(distances, expected_host=True, name="distances")

    cdef cydlpack.DLManagedTensor* queries_dlpack = \
        cydlpack.dlpack_c(queries_ai)
    cdef cydlpack.DLManagedTensor* neighbors_dlpack = \
        cydlpack.dlpack_c(neighbors_ai)
    cdef cydlpack.DLManagedTensor* distances_dlpack = \
        cydlpack.dlpack_c(distances_ai)

    # Perform search
    with cuda_interruptible():
        check_cuvs(cuvsMultiGpuIvfPqSearch(res, search_params.mg_params,
                                           index.mg_index, queries_dlpack,
                                           neighbors_dlpack, distances_dlpack))

    return (distances, neighbors)


@auto_sync_multi_gpu_resources
def extend(MultiGpuIndex index, new_vectors, new_indices=None,
           resources=None):
    """
    Extend the multi-GPU IVF-PQ index with new vectors.

    Parameters
    ----------
    index : :py:class:`cuvs.neighbors.mg_ivf_pq.MultiGpuIndex`
    new_vectors : Array interface compliant matrix shape (n_new_vectors, dim)
        Supported dtype [float32, float16, int8, uint8]
        **IMPORTANT**: For multi-GPU IVF-PQ, new_vectors MUST be
        in host memory (CPU).
        If using CuPy/device arrays, transfer to host with array.get()
        or cp.asnumpy(array).
    new_indices : Array interface compliant matrix shape (n_new_vectors,)
        , optional
        If provided, these indices will be used for the new vectors.
        If not provided, indices will be automatically assigned.
        **IMPORTANT**: Must be in host memory (CPU) for multi-GPU IVF-PQ.
    {resources_docstring}

    Examples
    --------

    >>> import numpy as np
    >>> from cuvs.neighbors import mg_ivf_pq
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_new_vectors = 1000
    >>> # For multi-GPU IVF-PQ, use host (NumPy) arrays
    >>> dataset = np.random.random_sample((n_samples, n_features),
    ...                                   dtype=np.float32)
    >>> new_vectors = np.random.random_sample((n_new_vectors, n_features),
    ...                                        dtype=np.float32)
    >>> build_params = mg_ivf_pq.MultiGpuIndexParams(metric="sqeuclidean")
    >>> index = mg_ivf_pq.build(build_params, dataset)
    >>> mg_ivf_pq.extend(index, new_vectors)
    """

    if not index.trained:
        raise ValueError("Index needs to be built before extending")

    new_vectors_ai = wrap_array(new_vectors)
    _check_input_array(new_vectors_ai,
                       [np.dtype('float32'), np.dtype('float16'),
                        np.dtype('byte'), np.dtype('ubyte')])

    # Multi-GPU IVF-PQ requires new_vectors in host memory
    _check_memory_location(new_vectors, expected_host=True, name="new_vectors")

    # Get resources
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef cydlpack.DLManagedTensor* new_vectors_dlpack = \
        cydlpack.dlpack_c(new_vectors_ai)
    cdef cydlpack.DLManagedTensor* new_indices_dlpack = NULL

    if new_indices is not None:
        new_indices_ai = wrap_array(new_indices)
        _check_input_array(new_indices_ai, [np.dtype('int64')])
        # Multi-GPU IVF-PQ requires new_indices in host memory
        _check_memory_location(new_indices, expected_host=True,
                               name="new_indices")
        new_indices_dlpack = cydlpack.dlpack_c(new_indices_ai)

    with cuda_interruptible():
        check_cuvs(cuvsMultiGpuIvfPqExtend(res, index.mg_index,
                                           new_vectors_dlpack,
                                           new_indices_dlpack))


@auto_sync_multi_gpu_resources
def save(MultiGpuIndex index, filename, resources=None):
    """
    Serialize the multi-GPU IVF-PQ index to a file.

    Parameters
    ----------
    index : :py:class:`cuvs.neighbors.mg_ivf_pq.MultiGpuIndex`
    filename : str
        The filename to serialize the index to.
    {resources_docstring}

    Examples
    --------

    >>> import numpy as np
    >>> from cuvs.neighbors import mg_ivf_pq
    >>> n_samples = 50000
    >>> n_features = 50
    >>> # For multi-GPU IVF-PQ, use host (NumPy) arrays
    >>> dataset = np.random.random_sample((n_samples, n_features),
    ...                                   dtype=np.float32)
    >>> build_params = mg_ivf_pq.MultiGpuIndexParams(metric="sqeuclidean")
    >>> index = mg_ivf_pq.build(build_params, dataset)
    >>> mg_ivf_pq.save(index, "index.bin")
    """

    if not index.trained:
        raise ValueError("Index needs to be built before serializing")

    # Get resources
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef string filename_str = filename.encode('utf-8')
    check_cuvs(cuvsMultiGpuIvfPqSerialize(res, index.mg_index,
                                          filename_str.c_str()))


@auto_sync_multi_gpu_resources
def load(filename, resources=None):
    """
    Deserialize the multi-GPU IVF-PQ index from a file.

    Parameters
    ----------
    filename : str
        The filename to deserialize the index from.
    {resources_docstring}

    Returns
    -------
    index : MultiGpuIndex
        The deserialized index.

    Examples
    --------

    >>> from cuvs.neighbors import mg_ivf_pq
    >>> index = mg_ivf_pq.load("index.bin")
    """

    cdef MultiGpuIndex index = MultiGpuIndex()
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef string filename_str = filename.encode('utf-8')
    check_cuvs(cuvsMultiGpuIvfPqDeserialize(res, filename_str.c_str(),
                                            index.mg_index))
    index.mg_trained = True

    return index


@auto_sync_multi_gpu_resources
def distribute(filename, resources=None):
    """
    Distribute the multi-GPU IVF-PQ index across multiple GPUs from a file.

    Parameters
    ----------
    filename : str
        The filename to distribute the index from.
    {resources_docstring}

    Returns
    -------
    index : MultiGpuIndex
        The distributed index.

    Examples
    --------

    >>> from cuvs.neighbors import mg_ivf_pq
    >>> index = mg_ivf_pq.distribute("index.bin")
    """

    cdef MultiGpuIndex index = MultiGpuIndex()
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef string filename_str = filename.encode('utf-8')
    check_cuvs(cuvsMultiGpuIvfPqDistribute(res, filename_str.c_str(),
                                           index.mg_index))
    index.mg_trained = True

    return index
