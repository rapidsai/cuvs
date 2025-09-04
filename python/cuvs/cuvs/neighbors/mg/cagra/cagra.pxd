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

from libc.stdint cimport uint32_t
from libcpp cimport bool

# Import base single-GPU extension module for subclassing
cimport cuvs.neighbors.cagra.cagra as _cagra
from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLManagedTensor
from cuvs.neighbors.cagra.cagra cimport (
    IndexParams as SingleGpuIndexParams,
    SearchParams as SingleGpuSearchParams,
    cuvsCagraIndexParams_t,
    cuvsCagraSearchParams_t,
)


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

# Multi-GPU CAGRA structures and functions
cdef extern from "cuvs/neighbors/mg_cagra.h" nogil:

    cdef struct cuvsMultiGpuCagraIndexParams:
        cuvsCagraIndexParams_t base_params
        cuvsMultiGpuDistributionMode mode

    cdef struct cuvsMultiGpuCagraSearchParams:
        cuvsCagraSearchParams_t base_params
        cuvsMultiGpuReplicatedSearchMode search_mode
        cuvsMultiGpuShardedMergeMode merge_mode
        uint32_t n_rows_per_batch

    cdef struct cuvsMultiGpuCagraIndex:
        pass

    ctypedef cuvsMultiGpuCagraIndexParams* cuvsMultiGpuCagraIndexParams_t
    ctypedef cuvsMultiGpuCagraSearchParams* cuvsMultiGpuCagraSearchParams_t
    ctypedef cuvsMultiGpuCagraIndex* cuvsMultiGpuCagraIndex_t

    cuvsError_t cuvsMultiGpuCagraIndexParamsCreate(
        cuvsMultiGpuCagraIndexParams_t* index_params)

    cuvsError_t cuvsMultiGpuCagraIndexParamsDestroy(
        cuvsMultiGpuCagraIndexParams_t index_params)

    cuvsError_t cuvsMultiGpuCagraSearchParamsCreate(
        cuvsMultiGpuCagraSearchParams_t* params)

    cuvsError_t cuvsMultiGpuCagraSearchParamsDestroy(
        cuvsMultiGpuCagraSearchParams_t params)

    cuvsError_t cuvsMultiGpuCagraIndexCreate(cuvsMultiGpuCagraIndex_t* index)

    cuvsError_t cuvsMultiGpuCagraIndexDestroy(cuvsMultiGpuCagraIndex_t index)

    cuvsError_t cuvsMultiGpuCagraBuild(cuvsResources_t res,
                                       cuvsMultiGpuCagraIndexParams_t params,
                                       DLManagedTensor* dataset_tensor,
                                       cuvsMultiGpuCagraIndex_t index) except +

    cuvsError_t cuvsMultiGpuCagraSearch(
        cuvsResources_t res,
        cuvsMultiGpuCagraSearchParams_t params,
        cuvsMultiGpuCagraIndex_t index,
        DLManagedTensor* queries_tensor,
        DLManagedTensor* neighbors_tensor,
        DLManagedTensor* distances_tensor) except +

    cuvsError_t cuvsMultiGpuCagraSerialize(
        cuvsResources_t res,
        cuvsMultiGpuCagraIndex_t index,
        const char* filename) except +

    cuvsError_t cuvsMultiGpuCagraDeserialize(
        cuvsResources_t res,
        const char* filename,
        cuvsMultiGpuCagraIndex_t index) except +

    cuvsError_t cuvsMultiGpuCagraDistribute(
        cuvsResources_t res,
        const char* filename,
        cuvsMultiGpuCagraIndex_t index) except +

    cuvsError_t cuvsMultiGpuCagraExtend(
        cuvsResources_t res,
        cuvsMultiGpuCagraIndex_t index,
        DLManagedTensor* new_vectors_tensor,
        DLManagedTensor* new_indices_tensor) except +


cdef class IndexParams(SingleGpuIndexParams):
    cdef cuvsMultiGpuCagraIndexParams_t mg_params

cdef class SearchParams(SingleGpuSearchParams):
    cdef cuvsMultiGpuCagraSearchParams_t mg_params

cdef class Index:
    cdef cuvsMultiGpuCagraIndex_t mg_index
    cdef bool mg_trained
