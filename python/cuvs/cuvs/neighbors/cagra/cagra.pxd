#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

from libc.stdint cimport (
    int8_t,
    int32_t,
    int64_t,
    uint8_t,
    uint32_t,
    uint64_t,
    uintptr_t,
)
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType
from cuvs.neighbors.filters.filters cimport cuvsFilter
from cuvs.neighbors.ivf_pq.ivf_pq cimport (
    cuvsIvfPqIndexParams_t,
    cuvsIvfPqSearchParams_t,
)


cdef extern from "library_types.h":
    ctypedef enum cudaDataType_t:
        CUDA_R_32F "CUDA_R_32F"  # float
        CUDA_R_16F "CUDA_R_16F"  # half

        # uint8 - used to refer to IVF-PQ's fp8 storage type
        CUDA_R_8U "CUDA_R_8U"
        CUDA_R_8I "CUDA_R_8I"

cdef extern from "cuvs/neighbors/cagra.h" nogil:

    ctypedef enum cuvsCagraGraphBuildAlgo:
        IVF_PQ
        NN_DESCENT
        ITERATIVE_CAGRA_SEARCH

    ctypedef struct cuvsCagraCompressionParams:
        uint32_t pq_bits
        uint32_t pq_dim
        uint32_t vq_n_centers
        uint32_t kmeans_n_iters
        double vq_kmeans_trainset_fraction
        double pq_kmeans_trainset_fraction

    ctypedef cuvsCagraCompressionParams* cuvsCagraCompressionParams_t

    ctypedef struct cuvsIvfPqParams:
        cuvsIvfPqIndexParams_t ivf_pq_build_params
        cuvsIvfPqSearchParams_t ivf_pq_search_params
        float refinement_rate
    ctypedef cuvsIvfPqParams* cuvsIvfPqParams_t

    ctypedef struct cuvsCagraIndexParams:
        cuvsDistanceType metric
        size_t intermediate_graph_degree
        size_t graph_degree
        cuvsCagraGraphBuildAlgo build_algo
        size_t nn_descent_niter
        cuvsCagraCompressionParams_t compression
        cuvsIvfPqParams_t graph_build_params

    ctypedef cuvsCagraIndexParams* cuvsCagraIndexParams_t

    ctypedef enum cuvsCagraSearchAlgo:
        SINGLE_CTA,
        MULTI_CTA,
        MULTI_KERNEL,
        AUTO

    ctypedef enum cuvsCagraHashMode:
        HASH,
        SMALL,
        AUTO_HASH

    ctypedef struct cuvsCagraSearchParams:
        size_t max_queries
        size_t itopk_size
        size_t max_iterations
        cuvsCagraSearchAlgo algo
        size_t team_size
        size_t search_width
        size_t min_iterations
        size_t thread_block_size
        cuvsCagraHashMode hashmap_mode
        size_t hashmap_min_bitlen
        float hashmap_max_fill_rate
        uint32_t num_random_samplings
        uint64_t rand_xor_mask
        bool persistent
        float persistent_lifetime
        float persistent_device_usage

    ctypedef cuvsCagraSearchParams* cuvsCagraSearchParams_t

    ctypedef struct cuvsCagraIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsCagraIndex* cuvsCagraIndex_t

    cuvsError_t cuvsCagraCompressionParamsCreate(
        cuvsCagraCompressionParams_t* params)

    cuvsError_t cuvsCagraCompressionParamsDestroy(
        cuvsCagraCompressionParams_t index)

    cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t* params)

    cuvsError_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t index)

    cuvsError_t cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t* params)

    cuvsError_t cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t index)

    cuvsError_t cuvsCagraIndexCreate(cuvsCagraIndex_t* index)

    cuvsError_t cuvsCagraIndexDestroy(cuvsCagraIndex_t index)

    cuvsError_t cuvsCagraIndexGetDims(cuvsCagraIndex_t index, int64_t* dim)
    cuvsError_t cuvsCagraIndexGetSize(cuvsCagraIndex_t index, int64_t* size)
    cuvsError_t cuvsCagraIndexGetGraphDegree(cuvsCagraIndex_t index,
                                             int64_t* degree)
    cuvsError_t cuvsCagraIndexGetGraph(cuvsCagraIndex_t index,
                                       DLManagedTensor * graph)
    cuvsError_t cuvsCagraIndexGetDataset(cuvsCagraIndex_t index,
                                         DLManagedTensor * dataset)

    cuvsError_t cuvsCagraBuild(cuvsResources_t res,
                               cuvsCagraIndexParams* params,
                               DLManagedTensor* dataset,
                               cuvsCagraIndex_t index)

    cuvsError_t cuvsCagraSearch(cuvsResources_t res,
                                cuvsCagraSearchParams* params,
                                cuvsCagraIndex_t index,
                                DLManagedTensor* queries,
                                DLManagedTensor* neighbors,
                                DLManagedTensor* distances,
                                cuvsFilter filter)

    cuvsError_t cuvsCagraSerialize(cuvsResources_t res,
                                   const char * filename,
                                   cuvsCagraIndex_t index,
                                   bool include_dataset)

    cuvsError_t cuvsCagraSerializeToHnswlib(cuvsResources_t res,
                                            const char * filename,
                                            cuvsCagraIndex_t index)

    cuvsError_t cuvsCagraDeserialize(cuvsResources_t res,
                                     const char * filename,
                                     cuvsCagraIndex_t index)

    cuvsError_t cuvsCagraIndexFromArgs(cuvsResources_t res,
                                       cuvsDistanceType metric,
                                       DLManagedTensor * graph,
                                       DLManagedTensor * dataset,
                                       cuvsCagraIndex_t index)

    ctypedef struct cuvsCagraExtendParams:
        uint32_t max_chunk_size

    ctypedef cuvsCagraExtendParams* cuvsCagraExtendParams_t

    cuvsError_t cuvsCagraExtendParamsCreate(cuvsCagraExtendParams_t* params)
    cuvsError_t cuvsCagraExtendParamsDestroy(cuvsCagraExtendParams_t params)
    cuvsError_t cuvsCagraExtend(cuvsResources_t res,
                                cuvsCagraExtendParams_t params,
                                DLManagedTensor* additional_dataset,
                                cuvsCagraIndex_t index)


cdef class Index:
    """
    CAGRA index object. This object stores the trained CAGRA index state
    which can be used to perform nearest neighbors searches.
    """

    cdef cuvsCagraIndex_t index
    cdef bool trained
    cdef str active_index_type


cdef class IndexParams:
    cdef cuvsCagraIndexParams* params
    cdef public object compression
    cdef public object ivf_pq_build_params
    cdef public object ivf_pq_search_params

cdef class SearchParams:
    cdef cuvsCagraSearchParams * params
