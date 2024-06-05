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
from libc.stdint cimport uint32_t
from libcpp cimport bool

from cuvs.common cimport cydlpack
from cuvs.distance_type cimport cuvsDistanceType

from pylibraft.common import auto_convert_output, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible
from pylibraft.neighbors.common import _check_input_array

from cuvs.distance import DISTANCE_TYPES

from cuvs.common.c_api cimport cuvsResources_t

from cuvs.common.exceptions import check_cuvs


@auto_sync_resources
@auto_convert_output
def refine(dataset,
           queries,
           candidates,
           k=None,
           metric="sqeuclidean",
           indices=None,
           distances=None,
           resources=None):
    """
    Refine nearest neighbor search.

    Refinement is an operation that follows an approximate NN search. The
    approximate search has already selected n_candidates neighbor candidates
    for each query. We narrow it down to k neighbors. For each query, we
    calculate the exact distance between the query and its n_candidates
    neighbor candidate, and select the k nearest ones.

    Input arrays can be either CUDA array interface compliant matrices or
    array interface compliant matrices in host memory. All array must be in
    the same memory space.

    Parameters
    ----------
    dataset : array interface compliant matrix, shape (n_samples, dim)
        Supported dtype [float32, int8, uint8, float16]
    queries : array interface compliant matrix, shape (n_queries, dim)
        Supported dtype [float32, int8, uint8, float16]
    candidates : array interface compliant matrix, shape (n_queries, k0)
        Supported dtype int64
    k : int
        Number of neighbors to search (k <= k0). Optional if indices or
        distances arrays are given (in which case their second dimension
        is k).
    metric : str
        Name of distance metric to use, default ="sqeuclidean"
    indices :  Optional array interface compliant matrix shape \
               (n_queries, k).
        If supplied, neighbor indices will be written here in-place.
        (default None). Supported dtype int64.
    distances :  Optional array interface compliant matrix shape \
                 (n_queries, k).
        If supplied, neighbor indices will be written here in-place.
        (default None) Supported dtype float.
    {resources_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.common import Resources
    >>> from cuvs.neighbors import ivf_pq, refine
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> resources = Resources()
    >>> index_params = ivf_pq.IndexParams(n_lists=1024,
    ...                                   metric="sqeuclidean",
    ...                                   pq_dim=10)
    >>> index = ivf_pq.build(index_params, dataset, resources=resources)
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 40
    >>> _, candidates = ivf_pq.search(ivf_pq.SearchParams(), index,
    ...                               queries, k, resources=resources)
    >>> k = 10
    >>> distances, neighbors = refine(dataset, queries, candidates, k)
    """
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    if k is None:
        if indices is not None:
            k = wrap_array(indices).shape[1]
        elif distances is not None:
            k = wrap_array(distances).shape[1]
        else:
            raise ValueError("Argument k must be specified if both indices "
                             "and distances arg is None")

    queries_cai = wrap_array(queries)
    dataset_cai = wrap_array(dataset)
    candidates_cai = wrap_array(candidates)
    n_queries = wrap_array(queries).shape[0]

    on_device = hasattr(dataset, "__cuda_array_interface__")
    ndarray = device_ndarray if on_device else np
    if indices is None:
        indices = ndarray.empty((n_queries, k), dtype='int64')

    if distances is None:
        distances = ndarray.empty((n_queries, k), dtype='float32')

    indices_cai = wrap_array(indices)
    distances_cai = wrap_array(distances)

    _check_input_array(indices_cai, [np.dtype('int64')],
                       exp_rows=n_queries, exp_cols=k)
    _check_input_array(distances_cai, [np.dtype('float32')],
                       exp_rows=n_queries, exp_cols=k)

    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_cai)
    cdef cydlpack.DLManagedTensor* queries_dlpack = \
        cydlpack.dlpack_c(queries_cai)
    cdef cydlpack.DLManagedTensor* candidates_dlpack = \
        cydlpack.dlpack_c(candidates_cai)
    cdef cydlpack.DLManagedTensor* indices_dlpack = \
        cydlpack.dlpack_c(indices_cai)
    cdef cydlpack.DLManagedTensor* distances_dlpack = \
        cydlpack.dlpack_c(distances_cai)

    cdef cuvsDistanceType c_metric = <cuvsDistanceType>DISTANCE_TYPES[metric]

    with cuda_interruptible():
        check_cuvs(cuvsRefine(
            res,
            dataset_dlpack,
            queries_dlpack,
            candidates_dlpack,
            c_metric,
            indices_dlpack,
            distances_dlpack))

    return (distances, indices)
