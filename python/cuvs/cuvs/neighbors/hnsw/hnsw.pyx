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

from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.string cimport string

from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources

from cuvs.common cimport cydlpack

import numpy as np

from cuvs.distance import DISTANCE_TYPES

from cuvs.neighbors.cagra cimport cagra

import os
import uuid

from pylibraft.common import auto_convert_output
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible
from pylibraft.neighbors.common import _check_input_array


cdef class SearchParams:
    """
    HNSW search parameters

    Parameters
    ----------
    ef: int, default = 200
        Maximum number of candidate list size used during search.
    num_threads: int, default = 0
        Number of CPU threads used to increase search parallelism.
        When set to 0, the number of threads is automatically determined
        using OpenMP's `omp_get_max_threads()`.
    """

    cdef cuvsHnswSearchParams params

    def __init__(self, *,
                 ef=200,
                 num_threads=0):
        self.params.ef = ef
        self.params.numThreads = num_threads

    def __repr__(self):
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in [
                        "ef", "num_threads"]]
        return "SearchParams(type=HNSW, " + (", ".join(attr_str)) + ")"

    @property
    def ef(self):
        return self.params.ef

    @property
    def num_threads(self):
        return self.params.numThreads


cdef class Index:
    """
    HNSW index object. This object stores the trained HNSW index state
    which can be used to perform nearest neighbors searches.
    """

    cdef cuvsHnswIndex_t index
    cdef bool trained

    def __cinit__(self):
        self.trained = False
        check_cuvs(cuvsHnswIndexCreate(&self.index))

    def __dealloc__(self):
        if self.index is not NULL:
            check_cuvs(cuvsHnswIndexDestroy(self.index))

    @property
    def trained(self):
        return self.trained

    def __repr__(self):
        # todo(dgd): update repr as we expose data through C API
        attr_str = []
        return "Index(type=HNSW, metric=L2" + (", ".join(attr_str)) + ")"


@auto_sync_resources
def save(filename, cagra.Index index, resources=None):
    """
    Saves the CAGRA index to a file as an hnswlib index.
    The saved index is immutable and can only be searched by the hnswlib
    wrapper in cuVS, as the format is not compatible with the original
    hnswlib.

    Saving / loading the index is experimental. The serialization format is
    subject to change.

    Parameters
    ----------
    filename : string
        Name of the file.
    index : Index
        Trained CAGRA index.
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
    >>> hnsw.save("my_index.bin", index)
    """
    cdef string c_filename = filename.encode('utf-8')
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    check_cuvs(cagra.cuvsCagraSerializeToHnswlib(res,
                                                 c_filename.c_str(),
                                                 index.index))


@auto_sync_resources
def load(filename, dim, dtype, metric="sqeuclidean", resources=None):
    """
    Loads base-layer-only hnswlib index from file, which was originally
    saved as a built CAGRA index. The loaded index is immutable and can only
    be searched by the hnswlib wrapper in cuVS, as the format is not
    compatible with the original hnswlib.

    Saving / loading the index is experimental. The serialization format is
    subject to change, therefore loading an index saved with a previous
    version of cuVS is not guaranteed to work.

    Parameters
    ----------
    filename : string
        Name of the file.
    dim : int
        Dimensions of the training dataest
    dtype : np.dtype of the saved index
        Valid values for dtype: [np.float32, np.byte, np.ubyte]
    metric : string denoting the metric type, default="sqeuclidean"
        Valid values for metric: ["sqeuclidean", "inner_product"], where
            - sqeuclidean is the euclidean distance without the square root
              operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2,
            - inner_product distance is defined as
              distance(a, b) = \\sum_i a_i * b_i.
    {resources_docstring}

    Returns
    -------
    index : HnswIndex

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.neighbors import cagra
    >>> from cuvs.neighbors import hnsw
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> index = cagra.build(cagra.IndexParams(), dataset)
    >>> # Serialize the CAGRA index to hnswlib base layer only index format
    >>> hnsw.save("my_index.bin", index)
    >>> index = hnsw.load("my_index.bin", n_features, np.float32,
    ...                   "sqeuclidean")
    """
    cdef Index idx = Index()
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef string c_filename = filename.encode('utf-8')
    cdef cydlpack.DLDataType dl_dtype
    if dtype == np.float32:
        dl_dtype.code = cydlpack.kDLFloat
        dl_dtype.bits = 32
        dl_dtype.lanes = 1
    elif dtype == np.ubyte:
        dl_dtype.code = cydlpack.kDLUInt
        dl_dtype.bits = 8
        dl_dtype.lanes = 1
    elif dtype == np.byte:
        dl_dtype.code = cydlpack.kDLInt
        dl_dtype.bits = 8
        dl_dtype.lanes = 1
    else:
        raise ValueError("Only float32 is supported for dtype")

    idx.index.dtype = dl_dtype
    cdef cuvsDistanceType distance_type = DISTANCE_TYPES[metric]

    check_cuvs(cuvsHnswDeserialize(
        res,
        c_filename.c_str(),
        dim,
        distance_type,
        idx.index
    ))
    idx.trained = True
    return idx


@auto_sync_resources
def from_cagra(cagra.Index index, temporary_index_path=None, resources=None):
    """
    Returns an hnsw base-layer-only index from a CAGRA index.

    NOTE: This method uses the filesystem to write the CAGRA index in
          `/tmp/<random_number>.bin` or the parameter `temporary_index_path`
          if not None before reading it as an hnsw index,
          then deleting the temporary file. The returned index is immutable
          and can only be searched by the hnsw wrapper in cuVS, as the
          format is not compatible with the original hnswlib library.
          By `base_layer_only`, we mean that the hnsw index is created
          without the additional layers that are used for the hierarchical
          search in hnswlib. Instead, the base layer is used for the search.

    Saving / loading the index is experimental. The serialization format is
    subject to change.

    Parameters
    ----------
    index : Index
        Trained CAGRA index.
    temporary_index_path : string, default = None
        Path to save the temporary index file. If None, the temporary file
        will be saved in `/tmp/<random_number>.bin`.
    {resources_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.neighbors import cagra
    >>> from cuvs.neighbors import hnsw
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> index = cagra.build(cagra.IndexParams(), dataset)
    >>> # Serialize the CAGRA index to hnswlib base layer only index format
    >>> hnsw_index = hnsw.from_cagra(index)
    """
    uuid_num = uuid.uuid4()
    filename = temporary_index_path if temporary_index_path else \
        f"/tmp/{uuid_num}.bin"
    save(filename, index, resources=resources)
    hnsw_index = load(filename, index.dim, np.dtype(index.active_index_type),
                      "sqeuclidean", resources=resources)
    os.remove(filename)
    return hnsw_index


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
        Supported dtype [float, int]
    k : int
        The number of neighbors.
    neighbors : Optional CUDA array interface compliant matrix shape
                (n_queries, k), dtype uint64_t. If supplied, neighbor
                indices will be written here in-place. (default None)
    distances : Optional CUDA array interface compliant matrix shape
                (n_queries, k) If supplied, the distances to the
                neighbors will be written here in-place. (default None)
    {resources_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.neighbors import cagra, hnsw
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
    >>> search_params = hnsw.SearchParams(
    ...     ef=200,
    ...     num_threads=0
    ... )
    >>> # Convert CAGRA index to HNSW
    >>> hnsw_index = hnsw.from_cagra(index)
    >>> # Using a pooling allocator reduces overhead of temporary array
    >>> # creation during search. This is useful if multiple searches
    >>> # are performed with same query size.
    >>> distances, neighbors = hnsw.search(search_params, index, queries,
    ...                                     k)
    >>> neighbors = cp.asarray(neighbors)
    >>> distances = cp.asarray(distances)
    """
    if not index.trained:
        raise ValueError("Index needs to be built before calling search.")

    # todo(dgd): we can make the check of dtype a parameter of wrap_array
    # in RAFT to make this a single call
    queries_ai = wrap_array(queries)
    _check_input_array(queries_ai, [np.dtype('float32'),
                                    np.dtype('uint8'),
                                    np.dtype('int8')])

    cdef uint32_t n_queries = queries_ai.shape[0]

    if neighbors is None:
        neighbors = np.empty((n_queries, k), dtype='uint64')

    neighbors_ai = wrap_array(neighbors)
    _check_input_array(neighbors_ai, [np.dtype('uint64')],
                       exp_rows=n_queries, exp_cols=k)

    if distances is None:
        distances = np.empty((n_queries, k), dtype='float32')

    distances_ai = wrap_array(distances)
    _check_input_array(distances_ai, [np.dtype('float32')],
                       exp_rows=n_queries, exp_cols=k)

    cdef cuvsHnswSearchParams* params = &search_params.params
    cdef cydlpack.DLManagedTensor* queries_dlpack = \
        cydlpack.dlpack_c(queries_ai)
    cdef cydlpack.DLManagedTensor* neighbors_dlpack = \
        cydlpack.dlpack_c(neighbors_ai)
    cdef cydlpack.DLManagedTensor* distances_dlpack = \
        cydlpack.dlpack_c(distances_ai)
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        check_cuvs(cuvsHnswSearch(
            res,
            params,
            index.index,
            queries_dlpack,
            neighbors_dlpack,
            distances_dlpack
        ))

    return (distances, neighbors)
