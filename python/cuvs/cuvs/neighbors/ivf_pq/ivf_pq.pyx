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
from pylibraft.neighbors.common import _check_input_array

from cuvs.distance import DISTANCE_TYPES

from libc.stdint cimport (
    int8_t,
    int64_t,
    uint8_t,
    uint32_t,
    uint64_t,
    uintptr_t,
)

from cuvs.common.exceptions import check_cuvs


cdef class IndexParams:
    """
    Parameters to build index for IvfPq nearest neighbor search

    Parameters
    ----------
    n_lists : int, default = 1024
        The number of clusters used in the coarse quantizer.
    metric : str, default="sqeuclidean"
        String denoting the metric type.
        Valid values for metric: ["sqeuclidean", "inner_product", "euclidean"],
        where:
            - sqeuclidean is the euclidean distance without the square root
              operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2,
            - euclidean is the euclidean distance
            - inner product distance is defined as
              distance(a, b) = \\sum_i a_i * b_i.
    kmeans_n_iters : int, default = 20
        The number of iterations searching for kmeans centers during index
        building.
    kmeans_trainset_fraction : int, default = 0.5
        If kmeans_trainset_fraction is less than 1, then the dataset is
        subsampled, and only n_samples * kmeans_trainset_fraction rows
        are used for training.
    pq_bits : int, default = 8
        The bit length of the vector element after quantization.
    pq_dim : int, default = 0
        The dimensionality of a the vector after product quantization.
        When zero, an optimal value is selected using a heuristic. Note
        pq_dim * pq_bits must be a multiple of 8. Hint: a smaller 'pq_dim'
        results in a smaller index size and better search performance, but
        lower recall. If 'pq_bits' is 8, 'pq_dim' can be set to any number,
        but multiple of 8 are desirable for good performance. If 'pq_bits'
        is not 8, 'pq_dim' should be a multiple of 8. For good performance,
        it is desirable that 'pq_dim' is a multiple of 32. Ideally,
        'pq_dim' should be also a divisor of the dataset dim.
    codebook_kind : string, default = "subspace"
        Valid values ["subspace", "cluster"]
    force_random_rotation : bool, default = False
        Apply a random rotation matrix on the input data and queries even
        if `dim % pq_dim == 0`. Note: if `dim` is not multiple of `pq_dim`,
        a random rotation is always applied to the input data and queries
        to transform the working space from `dim` to `rot_dim`, which may
        be slightly larger than the original space and and is a multiple
        of `pq_dim` (`rot_dim % pq_dim == 0`). However, this transform is
        not necessary when `dim` is multiple of `pq_dim` (`dim == rot_dim`,
        hence no need in adding "extra" data columns / features). By
        default, if `dim == rot_dim`, the rotation transform is
        initialized with the identity matrix. When
        `force_random_rotation == True`, a random orthogonal transform
        matrix is generated regardless of the values of `dim` and `pq_dim`.
    add_data_on_build : bool, default = True
        After training the coarse and fine quantizers, we will populate
        the index with the dataset if add_data_on_build == True, otherwise
        the index is left empty, and the extend method can be used
        to add new vectors to the index.
    conservative_memory_allocation : bool, default = True
        By default, the algorithm allocates more space than necessary for
        individual clusters (`list_data`). This allows to amortize the cost
        of memory allocation and reduce the number of data copies during
        repeated calls to `extend` (extending the database).
        To disable this behavior and use as little GPU memory for the
        database as possible, set this flat to `True`.
    max_train_points_per_pq_code : int, default = 256
        The max number of data points to use per PQ code during PQ codebook
        training. Using more data points per PQ code may increase the
        quality of PQ codebook but may also increase the build time. The
        parameter is applied to both PQ codebook generation methods, i.e.,
        PER_SUBSPACE and PER_CLUSTER. In both cases, we will use
        pq_book_size * max_train_points_per_pq_code training points to
        train each codebook.
    """

    cdef cuvsIvfPqIndexParams* params
    cdef object _metric

    def __cinit__(self):
        cuvsIvfPqIndexParamsCreate(&self.params)

    def __dealloc__(self):
        check_cuvs(cuvsIvfPqIndexParamsDestroy(self.params))

    def __init__(self, *,
                 n_lists=1024,
                 metric="sqeuclidean",
                 metric_arg=2.0,
                 kmeans_n_iters=20,
                 kmeans_trainset_fraction=0.5,
                 pq_bits=8,
                 pq_dim=0,
                 codebook_kind="subspace",
                 force_random_rotation=False,
                 add_data_on_build=True,
                 conservative_memory_allocation=False,
                 max_train_points_per_pq_code=256):
        self.params.n_lists = n_lists
        self._metric = metric
        self.params.metric = <cuvsDistanceType>DISTANCE_TYPES[metric]
        self.params.metric_arg = metric_arg
        self.params.kmeans_n_iters = kmeans_n_iters
        self.params.kmeans_trainset_fraction = kmeans_trainset_fraction
        self.params.pq_bits = pq_bits
        self.params.pq_dim = pq_dim
        if codebook_kind == "subspace":
            self.params.codebook_kind = codebook_gen.PER_SUBSPACE
        elif codebook_kind == "cluster":
            self.params.codebook_kind = codebook_gen.PER_CLUSTER
        else:
            raise ValueError("Incorrect codebook kind %s" % codebook_kind)
        self.params.force_random_rotation = force_random_rotation
        self.params.add_data_on_build = add_data_on_build
        self.params.conservative_memory_allocation = \
            conservative_memory_allocation
        self.params.max_train_points_per_pq_code = \
            max_train_points_per_pq_code

    @property
    def metric(self):
        return self._metric

    @property
    def metric_arg(self):
        return self.params.metric_arg

    @property
    def add_data_on_build(self):
        return self.params.add_data_on_build

    @property
    def n_lists(self):
        return self.params.n_lists

    @property
    def kmeans_n_iters(self):
        return self.params.kmeans_n_iters

    @property
    def kmeans_trainset_fraction(self):
        return self.params.kmeans_trainset_fraction

    @property
    def pq_bits(self):
        return self.params.pq_bits

    @property
    def pq_dim(self):
        return self.params.pq_dim

    @property
    def codebook_kind(self):
        return self.params.codebook_kind

    @property
    def force_random_rotation(self):
        return self.params.force_random_rotation

    @property
    def add_data_on_build(self):
        return self.params.add_data_on_build

    @property
    def conservative_memory_allocation(self):
        return self.params.conservative_memory_allocation

    @property
    def max_train_points_per_pq_code(self):
        return self.params.max_train_points_per_pq_code

cdef class Index:
    """
    IvfPq index object. This object stores the trained IvfPq index state
    which can be used to perform nearest neighbors searches.
    """

    cdef cuvsIvfPqIndex_t index
    cdef bool trained

    def __cinit__(self):
        self.trained = False
        check_cuvs(cuvsIvfPqIndexCreate(&self.index))

    def __dealloc__(self):
        check_cuvs(cuvsIvfPqIndexDestroy(self.index))

    @property
    def trained(self):
        return self.trained

    def __repr__(self):
        return "Index(type=IvfPq)"


@auto_sync_resources
def build(IndexParams index_params, dataset, resources=None):
    """
    Build the IvfPq index from the dataset for efficient search.

    Parameters
    ----------
    index_params : :py:class:`cuvs.neighbors.ivf_pq.IndexParams`
        Parameters on how to build the index
    dataset : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, int8, uint8]
    {resources_docstring}

    Returns
    -------
    index: :py:class:`cuvs.neighbors.ivf_pq.Index`

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.neighbors import ivf_pq
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> k = 10
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> build_params = ivf_pq.IndexParams(metric="sqeuclidean")
    >>> index = ivf_pq.build(build_params, dataset)
    >>> distances, neighbors = ivf_pq.search(ivf_pq.SearchParams(),
    ...                                        index, dataset,
    ...                                        k)
    >>> distances = cp.asarray(distances)
    >>> neighbors = cp.asarray(neighbors)
    """

    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype('float32'), np.dtype('byte'),
                                    np.dtype('ubyte')])

    cdef Index idx = Index()
    cdef cuvsError_t build_status
    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cuvsIvfPqIndexParams* params = index_params.params

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        check_cuvs(cuvsIvfPqBuild(
            res,
            params,
            dataset_dlpack,
            idx.index
        ))
        idx.trained = True

    return idx


cdef _map_dtype_np_to_cuda(dtype, supported_dtypes=None):
    if supported_dtypes is not None and dtype not in supported_dtypes:
        raise TypeError("Type %s is not supported" % str(dtype))
    return {np.float32: cudaDataType_t.CUDA_R_32F,
            np.float16: cudaDataType_t.CUDA_R_16F,
            np.uint8: cudaDataType_t.CUDA_R_8U}[dtype]


cdef class SearchParams:
    """
    Supplemental parameters to search IVF-Pq index

    Parameters
    ----------
    n_probes: int
        The number of clusters to search.
    lut_dtype: default = np.float32
        Data type of look up table to be created dynamically at search
        time. The use of low-precision types reduces the amount of shared
        memory required at search time, so fast shared memory kernels can
        be used even for datasets with large dimansionality. Note that
        the recall is slightly degraded when low-precision type is
        selected. Possible values [np.float32, np.float16, np.uint8]
    internal_distance_dtype: default = np.float32
        Storage data type for distance/similarity computation.
        Possible values [np.float32, np.float16]
    """

    cdef cuvsIvfPqSearchParams* params

    def __cinit__(self):
        cuvsIvfPqSearchParamsCreate(&self.params)

    def __dealloc__(self):
        check_cuvs(cuvsIvfPqSearchParamsDestroy(self.params))

    def __init__(self, *, n_probes=20, lut_dtype=np.float32,
                 internal_distance_dtype=np.float32):
        self.params.n_probes = n_probes
        self.params.lut_dtype = _map_dtype_np_to_cuda(lut_dtype)
        self.params.internal_distance_dtype = \
            _map_dtype_np_to_cuda(internal_distance_dtype)

    @property
    def n_probes(self):
        return self.params.n_probes

    @property
    def n_probes(self):
        return self.params.n_probes

    @property
    def lut_dtype(self):
        return self.params.lut_dtype

    @property
    def internal_distance_dtype(self):
        return self.params.internal_distance_dtype


@auto_sync_resources
@auto_convert_output
def search(SearchParams search_params,
           Index index,
           queries,
           k,
           neighbors=None,
           distances=None,
           resources=None):
    """
    Find the k nearest neighbors for each query.

    Parameters
    ----------
    search_params : :py:class:`cuvs.neighbors.ivf_pq.SearchParams`
        Parameters on how to search the index
    index : :py:class:`cuvs.neighbors.ivf_pq.Index`
        Trained IvfPq index.
    queries : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, int8, uint8]
    k : int
        The number of neighbors.
    neighbors : Optional CUDA array interface compliant matrix shape
                (n_queries, k), dtype int64_t. If supplied, neighbor
                indices will be written here in-place. (default None)
    distances : Optional CUDA array interface compliant matrix shape
                (n_queries, k) If supplied, the distances to the
                neighbors will be written here in-place. (default None)
    {resources_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.neighbors import ivf_pq
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build the index
    >>> index = ivf_pq.build(ivf_pq.IndexParams(), dataset)
    >>>
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 10
    >>> search_params = ivf_pq.SearchParams(n_probes=20)
    >>>
    >>> distances, neighbors = ivf_pq.search(search_params, index, queries,
    ...                                     k)
    """
    if not index.trained:
        raise ValueError("Index needs to be built before calling search.")

    queries_cai = wrap_array(queries)
    _check_input_array(queries_cai, [np.dtype('float32'), np.dtype('byte'),
                                     np.dtype('ubyte')])

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

    cdef cuvsIvfPqSearchParams* params = search_params.params
    cdef cuvsError_t search_status
    cdef cydlpack.DLManagedTensor* queries_dlpack = \
        cydlpack.dlpack_c(queries_cai)
    cdef cydlpack.DLManagedTensor* neighbors_dlpack = \
        cydlpack.dlpack_c(neighbors_cai)
    cdef cydlpack.DLManagedTensor* distances_dlpack = \
        cydlpack.dlpack_c(distances_cai)
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        check_cuvs(cuvsIvfPqSearch(
            res,
            params,
            index.index,
            queries_dlpack,
            neighbors_dlpack,
            distances_dlpack
        ))

    return (distances, neighbors)


@auto_sync_resources
def save(filename, Index index, bool include_dataset=True, resources=None):
    """
    Saves the index to a file.

    Saving / loading the index is experimental. The serialization format is
    subject to change.

    Parameters
    ----------
    filename : string
        Name of the file.
    index : Index
        Trained IVF-PQ index.
    {resources_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.neighbors import ivf_pq
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> index = ivf_pq.build(ivf_pq.IndexParams(), dataset)
    >>> # Serialize and deserialize the ivf_pq index built
    >>> ivf_pq.save("my_index.bin", index)
    >>> index_loaded = ivf_pq.load("my_index.bin")
    """
    cdef string c_filename = filename.encode('utf-8')
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    check_cuvs(cuvsIvfPqSerialize(res,
                                  c_filename.c_str(),
                                  index.index))


@auto_sync_resources
def load(filename, resources=None):
    """
    Loads index from file.

    Saving / loading the index is experimental. The serialization format is
    subject to change, therefore loading an index saved with a previous
    version of cuvs is not guaranteed to work.

    Parameters
    ----------
    filename : string
        Name of the file.
    {resources_docstring}

    Returns
    -------
    index : Index

    """
    cdef Index idx = Index()
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef string c_filename = filename.encode('utf-8')

    check_cuvs(cuvsIvfPqDeserialize(
        res,
        c_filename.c_str(),
        idx.index
    ))
    idx.trained = True
    return idx


@auto_sync_resources
def extend(Index index, new_vectors, new_indices, resources=None):
    """
    Extend an existing index with new vectors.

    The input array can be either CUDA array interface compliant matrix or
    array interface compliant matrix in host memory.


    Parameters
    ----------
    index : ivf_pq.Index
        Trained ivf_pq object.
    new_vectors : array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, int8, uint8]
    new_indices : array interface compliant vector shape (n_samples)
        Supported dtype [int64]
    {resources_docstring}

    Returns
    -------
    index: py:class:`cuvs.neighbors.ivf_pq.Index`

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.neighbors import ivf_pq
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> index = ivf_pq.build(ivf_pq.IndexParams(), dataset)
    >>> n_rows = 100
    >>> more_data = cp.random.random_sample((n_rows, n_features),
    ...                                     dtype=cp.float32)
    >>> indices = n_samples + cp.arange(n_rows, dtype=cp.int64)
    >>> index = ivf_pq.extend(index, more_data, indices)
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> distances, neighbors = ivf_pq.search(ivf_pq.SearchParams(),
    ...                                      index, queries,
    ...                                      k=10)
    """

    new_vectors_ai = wrap_array(new_vectors)
    _check_input_array(new_vectors_ai, [np.dtype('float32'), np.dtype('byte'),
                                        np.dtype('ubyte')])

    new_indices_ai = wrap_array(new_indices)
    _check_input_array(new_indices_ai, [np.dtype('int64')])
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef cydlpack.DLManagedTensor* new_vectors_dlpack = \
        cydlpack.dlpack_c(new_vectors_ai)

    cdef cydlpack.DLManagedTensor* new_indices_dlpack = \
        cydlpack.dlpack_c(new_indices_ai)

    with cuda_interruptible():
        check_cuvs(cuvsIvfPqExtend(
            res,
            new_vectors_dlpack,
            new_indices_dlpack,
            index.index
        ))

    return index
