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

import warnings

import numpy as np

cimport cuvs.common.cydlpack

from cuvs.common.resources import auto_sync_resources

from cython.operator cimport dereference as deref
from libcpp cimport bool, cast
from libcpp.string cimport string

from cuvs.common cimport cydlpack

from pylibraft.common import auto_convert_output, cai_wrapper, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible
from pylibraft.neighbors.common import _check_input_array

from libc.stdint cimport (
    int8_t,
    int64_t,
    uint8_t,
    uint32_t,
    uint64_t,
    uintptr_t,
)

from cuvs.common.exceptions import check_cuvs


cdef class CompressionParams:
    """
    Parameters for VPQ Compression

    Parameters
    ----------
    pq_bits: int
        The bit length of the vector element after compression by PQ.
        Possible values: [4, 5, 6, 7, 8]. The smaller the 'pq_bits', the
        smaller the index size and the better the search performance, but
        the lower the recall.
    pq_dim: int
        The dimensionality of the vector after compression by PQ. When zero,
        an optimal value is selected using a heuristic.
    vq_n_centers: int
        Vector Quantization (VQ) codebook size - number of "coarse cluster
        centers". When zero, an optimal value is selected using a heuristic.
    kmeans_n_iters: int
        The number of iterations searching for kmeans centers (both VQ & PQ
        phases).
    vq_kmeans_trainset_fraction: float
        The fraction of data to use during iterative kmeans building (VQ
        phase). When zero, an optimal value is selected using a heuristic.
    vq_kmeans_trainset_fraction: float
        The fraction of data to use during iterative kmeans building (PQ
        phase). When zero, an optimal value is selected using a heuristic.
    """
    cdef cuvsCagraCompressionParams * params

    def __cinit__(self):
        check_cuvs(cuvsCagraCompressionParamsCreate(&self.params))

    def __dealloc__(self):
        check_cuvs(cuvsCagraCompressionParamsDestroy(self.params))

    def __init__(self, *,
                 pq_bits=8,
                 pq_dim=0,
                 vq_n_centers=0,
                 kmeans_n_iters=25,
                 vq_kmeans_trainset_fraction=0.0,
                 pq_kmeans_trainset_fraction=0.0):
        self.params.pq_bits = pq_bits
        self.params.pq_dim = pq_dim
        self.params.vq_n_centers = vq_n_centers
        self.params.kmeans_n_iters = kmeans_n_iters
        self.params.vq_kmeans_trainset_fraction = vq_kmeans_trainset_fraction
        self.params.pq_kmeans_trainset_fraction = pq_kmeans_trainset_fraction

    @property
    def pq_bits(self):
        return self.params.pq_bits

    @property
    def pq_dim(self):
        return self.params.pq_dim

    @property
    def vq_n_centers(self):
        return self.params.vq_n_centers

    @property
    def kmeans_n_iters(self):
        return self.params.kmeans_n_iters

    @property
    def vq_kmeans_trainset_fraction(self):
        return self.params.vq_kmeans_trainset_fraction

    @property
    def pq_kmeans_trainset_fraction(self):
        return self.params.pq_kmeans_trainset_fraction

    def get_handle(self):
        return <size_t>self.params

cdef class IndexParams:
    """
    Parameters to build index for CAGRA nearest neighbor search

    Parameters
    ----------
    metric : string denoting the metric type, default="sqeuclidean"
        Valid values for metric: ["sqeuclidean"], where
            - sqeuclidean is the euclidean distance without the square root
              operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2
    intermediate_graph_degree : int, default = 128

    graph_degree : int, default = 64

    build_algo: string denoting the graph building algorithm to use, \
                default = "ivf_pq"
        Valid values for algo: ["ivf_pq", "nn_descent"], where
            - ivf_pq will use the IVF-PQ algorithm for building the knn graph
            - nn_descent (experimental) will use the NN-Descent algorithm for
              building the knn graph. It is expected to be generally
              faster than ivf_pq.
    compression: CompressionParams, optional
        If compression is desired should be a CompressionParams object. If None
        compression will be disabled.
    """

    cdef cuvsCagraIndexParams* params

    # hold on to a reference to the compression, to keep from being GC'ed
    cdef public object compression

    def __cinit__(self):
        check_cuvs(cuvsCagraIndexParamsCreate(&self.params))
        self.compression = None

    def __dealloc__(self):
        check_cuvs(cuvsCagraIndexParamsDestroy(self.params))

    def __init__(self, *,
                 metric="sqeuclidean",
                 intermediate_graph_degree=128,
                 graph_degree=64,
                 build_algo="ivf_pq",
                 nn_descent_niter=20,
                 compression=None):

        # todo (dgd): enable once other metrics are present
        # and exposed in cuVS C API
        # self.params.metric = _get_metric(metric)
        # self.params.metric_arg = 0
        self.params.intermediate_graph_degree = intermediate_graph_degree
        self.params.graph_degree = graph_degree
        if build_algo == "ivf_pq":
            self.params.build_algo = cuvsCagraGraphBuildAlgo.IVF_PQ
        elif build_algo == "nn_descent":
            self.params.build_algo = cuvsCagraGraphBuildAlgo.NN_DESCENT
        self.params.nn_descent_niter = nn_descent_niter
        if compression is not None:
            self.compression = compression
            self.params.compression = \
                <cuvsCagraCompressionParams_t><size_t>compression.get_handle()

    # @property
    # def metric(self):
        # return self.params.metric

    @property
    def intermediate_graph_degree(self):
        return self.params.intermediate_graph_degree

    @property
    def graph_degree(self):
        return self.params.graph_degree

    @property
    def build_algo(self):
        return self.params.build_algo

    @property
    def nn_descent_niter(self):
        return self.params.nn_descent_niter


cdef class Index:
    """
    CAGRA index object. This object stores the trained CAGRA index state
    which can be used to perform nearest neighbors searches.
    """

    cdef cuvsCagraIndex_t index
    cdef bool trained

    def __cinit__(self):
        self.trained = False
        check_cuvs(cuvsCagraIndexCreate(&self.index))

    def __dealloc__(self):
        if self.index is not NULL:
            check_cuvs(cuvsCagraIndexDestroy(self.index))

    @property
    def trained(self):
        return self.trained

    def __repr__(self):
        # todo(dgd): update repr as we expose data through C API
        attr_str = []
        return "Index(type=CAGRA, metric=L2" + (", ".join(attr_str)) + ")"


@auto_sync_resources
def build(IndexParams index_params, dataset, resources=None):
    """
    Build the CAGRA index from the dataset for efficient search.

    The build performs two different steps- first an intermediate knn-graph is
    constructed, then it's optimized it to create the final graph. The
    index_params object controls the node degree of these graphs.

    It is required that both the dataset and the optimized graph fit the
    GPU memory.

    The following distance metrics are supported:
        - L2

    Parameters
    ----------
    index_params : IndexParams object
    dataset : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, int8, uint8]
    {resources_docstring}

    Returns
    -------
    index: cuvs.cagra.Index

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.neighbors import cagra
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> k = 10
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> build_params = cagra.IndexParams(metric="sqeuclidean")
    >>> index = cagra.build(build_params, dataset)
    >>> distances, neighbors = cagra.search(cagra.SearchParams(),
    ...                                      index, dataset,
    ...                                      k)
    >>> distances = cp.asarray(distances)
    >>> neighbors = cp.asarray(neighbors)
    """

    # todo(dgd): we can make the check of dtype a parameter of wrap_array
    # in RAFT to make this a single call
    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype('float32'), np.dtype('byte'),
                                    np.dtype('ubyte')])

    cdef Index idx = Index()
    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cuvsCagraIndexParams* params = index_params.params

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        check_cuvs(cuvsCagraBuild(
            res,
            params,
            dataset_dlpack,
            idx.index
        ))
        idx.trained = True

    return idx


def build_index(IndexParams index_params, dataset, resources=None):
    warnings.warn("cagra.build_index is deprecated, use cagra.build instead",
                  FutureWarning)
    return build(index_params, dataset, resources=resources)


cdef class SearchParams:
    """
    CAGRA search parameters

    Parameters
    ----------
    max_queries: int, default = 0
        Maximum number of queries to search at the same time (batch size).
        Auto select when 0.
    itopk_size: int, default = 64
        Number of intermediate search results retained during the search.
        This is the main knob to adjust trade off between accuracy and
        search speed. Higher values improve the search accuracy.
    max_iterations: int, default = 0
        Upper limit of search iterations. Auto select when 0.
    algo: string denoting the search algorithm to use, default = "auto"
        Valid values for algo: ["auto", "single_cta", "multi_cta"], where
            - auto will automatically select the best value based on query size
            - single_cta is better when query contains larger number of
              vectors (e.g >10)
            - multi_cta is better when query contains only a few vectors
    team_size: int, default = 0
        Number of threads used to calculate a single distance. 4, 8, 16,
        or 32.
    search_width: int, default = 1
        Number of graph nodes to select as the starting point for the
        search in each iteration.
    min_iterations: int, default = 0
        Lower limit of search iterations.
    thread_block_size: int, default = 0
        Thread block size. 0, 64, 128, 256, 512, 1024.
        Auto selection when 0.
    hashmap_mode: string denoting the type of hash map to use.
        It's usually better to allow the algorithm to select this value,
        default = "auto".
        Valid values for hashmap_mode: ["auto", "small", "hash"], where
            - auto will automatically select the best value based on algo
            - small will use the small shared memory hash table with resetting.
            - hash will use a single hash table in global memory.
    hashmap_min_bitlen: int, default = 0
        Upper limit of hashmap fill rate. More than 0.1, less than 0.9.
    hashmap_max_fill_rate: float, default = 0.5
        Upper limit of hashmap fill rate. More than 0.1, less than 0.9.
    num_random_samplings: int, default = 1
        Number of iterations of initial random seed node selection. 1 or
        more.
    rand_xor_mask: int, default = 0x128394
        Bit mask used for initial random seed node selection.
    """

    cdef cuvsCagraSearchParams params

    def __init__(self, *,
                 max_queries=0,
                 itopk_size=64,
                 max_iterations=0,
                 algo="auto",
                 team_size=0,
                 search_width=1,
                 min_iterations=0,
                 thread_block_size=0,
                 hashmap_mode="auto",
                 hashmap_min_bitlen=0,
                 hashmap_max_fill_rate=0.5,
                 num_random_samplings=1,
                 rand_xor_mask=0x128394):
        self.params.max_queries = max_queries
        self.params.itopk_size = itopk_size
        self.params.max_iterations = max_iterations
        if algo == "single_cta":
            self.params.algo = cuvsCagraSearchAlgo.SINGLE_CTA
        elif algo == "multi_cta":
            self.params.algo = cuvsCagraSearchAlgo.MULTI_CTA
        elif algo == "multi_kernel":
            self.params.algo = cuvsCagraSearchAlgo.MULTI_KERNEL
        elif algo == "auto":
            self.params.algo = cuvsCagraSearchAlgo.AUTO
        else:
            raise ValueError("`algo` value not supported.")

        self.params.team_size = team_size
        self.params.search_width = search_width
        self.params.min_iterations = min_iterations
        self.params.thread_block_size = thread_block_size
        if hashmap_mode == "hash":
            self.params.hashmap_mode = cuvsCagraHashMode.HASH
        elif hashmap_mode == "small":
            self.params.hashmap_mode = cuvsCagraHashMode.SMALL
        elif hashmap_mode == "auto":
            self.params.hashmap_mode = cuvsCagraHashMode.AUTO_HASH
        else:
            raise ValueError("`hashmap_mode` value not supported.")

        self.params.hashmap_min_bitlen = hashmap_min_bitlen
        self.params.hashmap_max_fill_rate = hashmap_max_fill_rate
        self.params.num_random_samplings = num_random_samplings
        self.params.rand_xor_mask = rand_xor_mask

    def __repr__(self):
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in [
                        "max_queries", "itopk_size", "max_iterations", "algo",
                        "team_size", "search_width", "min_iterations",
                        "thread_block_size", "hashmap_mode",
                        "hashmap_min_bitlen", "hashmap_max_fill_rate",
                        "num_random_samplings", "rand_xor_mask"]]
        return "SearchParams(type=CAGRA, " + (", ".join(attr_str)) + ")"

    @property
    def max_queries(self):
        return self.params.max_queries

    @property
    def itopk_size(self):
        return self.params.itopk_size

    @property
    def max_iterations(self):
        return self.params.max_iterations

    @property
    def algo(self):
        return self.params.algo

    @property
    def team_size(self):
        return self.params.team_size

    @property
    def search_width(self):
        return self.params.search_width

    @property
    def min_iterations(self):
        return self.params.min_iterations

    @property
    def thread_block_size(self):
        return self.params.thread_block_size

    @property
    def hashmap_mode(self):
        return self.params.hashmap_mode

    @property
    def hashmap_min_bitlen(self):
        return self.params.hashmap_min_bitlen

    @property
    def hashmap_max_fill_rate(self):
        return self.params.hashmap_max_fill_rate

    @property
    def num_random_samplings(self):
        return self.params.num_random_samplings

    @property
    def rand_xor_mask(self):
        return self.params.rand_xor_mask


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
    search_params : SearchParams
    index : Index
        Trained CAGRA index.
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
    >>> from cuvs.neighbors import cagra
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> index = cagra.build(cagra.IndexParams(), dataset)
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 10
    >>> search_params = cagra.SearchParams(
    ...     max_queries=100,
    ...     itopk_size=64
    ... )
    >>> # Using a pooling allocator reduces overhead of temporary array
    >>> # creation during search. This is useful if multiple searches
    >>> # are performed with same query size.
    >>> distances, neighbors = cagra.search(search_params, index, queries,
    ...                                     k)
    >>> neighbors = cp.asarray(neighbors)
    >>> distances = cp.asarray(distances)
    """
    if not index.trained:
        raise ValueError("Index needs to be built before calling search.")

    # todo(dgd): we can make the check of dtype a parameter of wrap_array
    # in RAFT to make this a single call
    queries_cai = wrap_array(queries)
    _check_input_array(queries_cai, [np.dtype('float32'), np.dtype('byte'),
                                     np.dtype('ubyte')])

    cdef uint32_t n_queries = queries_cai.shape[0]

    if neighbors is None:
        neighbors = device_ndarray.empty((n_queries, k), dtype='uint32')

    neighbors_cai = wrap_array(neighbors)
    _check_input_array(neighbors_cai, [np.dtype('uint32')],
                       exp_rows=n_queries, exp_cols=k)

    if distances is None:
        distances = device_ndarray.empty((n_queries, k), dtype='float32')

    distances_cai = wrap_array(distances)
    _check_input_array(distances_cai, [np.dtype('float32')],
                       exp_rows=n_queries, exp_cols=k)

    cdef cuvsCagraSearchParams* params = &search_params.params
    cdef cydlpack.DLManagedTensor* queries_dlpack = \
        cydlpack.dlpack_c(queries_cai)
    cdef cydlpack.DLManagedTensor* neighbors_dlpack = \
        cydlpack.dlpack_c(neighbors_cai)
    cdef cydlpack.DLManagedTensor* distances_dlpack = \
        cydlpack.dlpack_c(distances_cai)
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        check_cuvs(cuvsCagraSearch(
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
        Trained CAGRA index.
    include_dataset : bool
        Whether or not to write out the dataset along with the index. Including
        the dataset in the serialized index will use extra disk space, and
        might not be desired if you already have a copy of the dataset on
        disk. If this option is set to false, you will have to call
        `index.update_dataset(dataset)` after loading the index.
    {resources_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.neighbors import cagra
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> index = cagra.build(cagra.IndexParams(), dataset)
    >>> # Serialize and deserialize the cagra index built
    >>> cagra.save("my_index.bin", index)
    >>> index_loaded = cagra.load("my_index.bin")
    """
    cdef string c_filename = filename.encode('utf-8')
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    check_cuvs(cuvsCagraSerialize(res,
                                  c_filename.c_str(),
                                  index.index,
                                  include_dataset))


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

    check_cuvs(cuvsCagraDeserialize(
        res,
        c_filename.c_str(),
        idx.index
    ))
    idx.trained = True
    return idx
