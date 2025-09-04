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

# Import base single-GPU extension module for subclassing
from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.neighbors.ivf_flat.ivf_flat cimport (
    IndexParams as SingleGpuIndexParams,
    SearchParams as SingleGpuSearchParams,
    cuvsIvfFlatIndexParams_t,
    cuvsIvfFlatSearchParams_t,
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

# Multi-GPU IVF-Flat structures and functions
cdef extern from "cuvs/neighbors/mg_ivf_flat.h" nogil:
    cdef struct cuvsMultiGpuIvfFlatIndexParams:
        cuvsIvfFlatIndexParams_t base_params
        cuvsMultiGpuDistributionMode mode

    cdef struct cuvsMultiGpuIvfFlatSearchParams:
        cuvsIvfFlatSearchParams_t base_params
        cuvsMultiGpuReplicatedSearchMode search_mode
        cuvsMultiGpuShardedMergeMode merge_mode
        int64_t n_rows_per_batch

    cdef struct cuvsMultiGpuIvfFlatIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsMultiGpuIvfFlatIndexParams* cuvsMultiGpuIvfFlatIndexParams_t
    ctypedef cuvsMultiGpuIvfFlatSearchParams* cuvsMultiGpuIvfFlatSearchParams_t
    ctypedef cuvsMultiGpuIvfFlatIndex* cuvsMultiGpuIvfFlatIndex_t

    cuvsError_t cuvsMultiGpuIvfFlatIndexParamsCreate(
        cuvsMultiGpuIvfFlatIndexParams_t* index_params)

    cuvsError_t cuvsMultiGpuIvfFlatIndexParamsDestroy(
        cuvsMultiGpuIvfFlatIndexParams_t index_params)

    cuvsError_t cuvsMultiGpuIvfFlatSearchParamsCreate(
        cuvsMultiGpuIvfFlatSearchParams_t* params)

    cuvsError_t cuvsMultiGpuIvfFlatSearchParamsDestroy(
        cuvsMultiGpuIvfFlatSearchParams_t params)

    cuvsError_t cuvsMultiGpuIvfFlatIndexCreate(
        cuvsMultiGpuIvfFlatIndex_t* index)

    cuvsError_t cuvsMultiGpuIvfFlatIndexDestroy(
        cuvsMultiGpuIvfFlatIndex_t index)

    cuvsError_t cuvsMultiGpuIvfFlatBuild(
        cuvsResources_t res,
        cuvsMultiGpuIvfFlatIndexParams_t params,
        DLManagedTensor* dataset_tensor,
        cuvsMultiGpuIvfFlatIndex_t index) except +

    cuvsError_t cuvsMultiGpuIvfFlatSearch(
        cuvsResources_t res,
        cuvsMultiGpuIvfFlatSearchParams_t params,
        cuvsMultiGpuIvfFlatIndex_t index,
        DLManagedTensor* queries_tensor,
        DLManagedTensor* neighbors_tensor,
        DLManagedTensor* distances_tensor) except +

    cuvsError_t cuvsMultiGpuIvfFlatExtend(
        cuvsResources_t res,
        cuvsMultiGpuIvfFlatIndex_t index,
        DLManagedTensor* new_vectors_tensor,
        DLManagedTensor* new_indices_tensor) except +

    cuvsError_t cuvsMultiGpuIvfFlatSerialize(
        cuvsResources_t res,
        cuvsMultiGpuIvfFlatIndex_t index,
        const char* filename) except +

    cuvsError_t cuvsMultiGpuIvfFlatDeserialize(
        cuvsResources_t res,
        const char* filename,
        cuvsMultiGpuIvfFlatIndex_t index) except +

    cuvsError_t cuvsMultiGpuIvfFlatDistribute(
        cuvsResources_t res,
        const char* filename,
        cuvsMultiGpuIvfFlatIndex_t index) except +


cdef class IndexParams(SingleGpuIndexParams):
    cdef cuvsMultiGpuIvfFlatIndexParams_t mg_params

cdef class SearchParams(SingleGpuSearchParams):
    cdef cuvsMultiGpuIvfFlatSearchParams_t mg_params

cdef class Index:
    cdef cuvsMultiGpuIvfFlatIndex_t mg_index
    cdef bool mg_trained
