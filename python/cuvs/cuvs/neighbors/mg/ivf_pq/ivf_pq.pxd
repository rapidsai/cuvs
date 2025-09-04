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

from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.neighbors.ivf_pq.ivf_pq cimport (
    IndexParams as SingleGpuIndexParams,
    SearchParams as SingleGpuSearchParams,
    cuvsIvfPqIndexParams_t,
    cuvsIvfPqSearchParams_t,
)

# Import base single-GPU extension module for subclassing

# Multi-GPU distribution modes
cdef extern from "cuvs/neighbors/mg_common.h" nogil:
    ctypedef enum cuvsMultiGpuDistributionMode:
        CUVS_NEIGHBORS_MG_REPLICATED
        CUVS_NEIGHBORS_MG_SHARDED

    ctypedef enum cuvsMultiGpuReplicatedSearchMode:
        CUVS_NEIGHBORS_MG_LOAD_BALANCER
        CUVS_NEIGHBORS_MG_ROUND_ROBIN

    ctypedef enum cuvsMultiGpuShardedMergeMode:
        CUVS_NEIGHBORS_MG_MERGE_ON_ROOT_RANK
        CUVS_NEIGHBORS_MG_TREE_MERGE

# Multi-GPU IVF-PQ structures and functions
cdef extern from "cuvs/neighbors/mg_ivf_pq.h" nogil:
    cdef struct cuvsMultiGpuIvfPqIndexParams:
        cuvsIvfPqIndexParams_t base_params
        cuvsMultiGpuDistributionMode mode

    cdef struct cuvsMultiGpuIvfPqSearchParams:
        cuvsIvfPqSearchParams_t base_params
        cuvsMultiGpuReplicatedSearchMode search_mode
        cuvsMultiGpuShardedMergeMode merge_mode
        int64_t n_rows_per_batch

    cdef struct cuvsMultiGpuIvfPqIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsMultiGpuIvfPqIndexParams* cuvsMultiGpuIvfPqIndexParams_t
    ctypedef cuvsMultiGpuIvfPqSearchParams* cuvsMultiGpuIvfPqSearchParams_t
    ctypedef cuvsMultiGpuIvfPqIndex* cuvsMultiGpuIvfPqIndex_t

    cuvsError_t cuvsMultiGpuIvfPqIndexParamsCreate(
        cuvsMultiGpuIvfPqIndexParams_t* index_params)

    cuvsError_t cuvsMultiGpuIvfPqIndexParamsDestroy(
        cuvsMultiGpuIvfPqIndexParams_t index_params)

    cuvsError_t cuvsMultiGpuIvfPqSearchParamsCreate(
        cuvsMultiGpuIvfPqSearchParams_t* params)

    cuvsError_t cuvsMultiGpuIvfPqSearchParamsDestroy(
        cuvsMultiGpuIvfPqSearchParams_t params)

    cuvsError_t cuvsMultiGpuIvfPqIndexCreate(cuvsMultiGpuIvfPqIndex_t* index)

    cuvsError_t cuvsMultiGpuIvfPqIndexDestroy(cuvsMultiGpuIvfPqIndex_t index)

    cuvsError_t cuvsMultiGpuIvfPqBuild(cuvsResources_t res,
                                       cuvsMultiGpuIvfPqIndexParams_t params,
                                       DLManagedTensor* dataset_tensor,
                                       cuvsMultiGpuIvfPqIndex_t index) except +

    cuvsError_t cuvsMultiGpuIvfPqSearch(
        cuvsResources_t res,
        cuvsMultiGpuIvfPqSearchParams_t params,
        cuvsMultiGpuIvfPqIndex_t index,
        DLManagedTensor* queries_tensor,
        DLManagedTensor* neighbors_tensor,
        DLManagedTensor* distances_tensor) except +

    cuvsError_t cuvsMultiGpuIvfPqExtend(
        cuvsResources_t res,
        cuvsMultiGpuIvfPqIndex_t index,
        DLManagedTensor* new_vectors_tensor,
        DLManagedTensor* new_indices_tensor) except +

    cuvsError_t cuvsMultiGpuIvfPqSerialize(
        cuvsResources_t res,
        cuvsMultiGpuIvfPqIndex_t index,
        const char* filename) except +

    cuvsError_t cuvsMultiGpuIvfPqDeserialize(
        cuvsResources_t res,
        const char* filename,
        cuvsMultiGpuIvfPqIndex_t index) except +

    cuvsError_t cuvsMultiGpuIvfPqDistribute(
        cuvsResources_t res,
        const char* filename,
        cuvsMultiGpuIvfPqIndex_t index) except +


cdef class IndexParams(SingleGpuIndexParams):
    cdef cuvsMultiGpuIvfPqIndexParams_t mg_params

cdef class SearchParams(SingleGpuSearchParams):
    cdef cuvsMultiGpuIvfPqSearchParams_t mg_params

cdef class Index:
    cdef cuvsMultiGpuIvfPqIndex_t mg_index
    cdef bool mg_trained
