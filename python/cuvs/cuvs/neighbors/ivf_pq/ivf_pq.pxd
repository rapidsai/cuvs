#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport int64_t, uint32_t, uintptr_t
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType


cdef extern from "library_types.h":
    ctypedef enum cudaDataType_t:
        CUDA_R_32F "CUDA_R_32F"  # float
        CUDA_R_16F "CUDA_R_16F"  # half

        # uint8 - used to refer to IVF-PQ's fp8 storage type
        CUDA_R_8U "CUDA_R_8U"
        CUDA_R_8I "CUDA_R_8I"


cdef extern from "cuvs/neighbors/ivf_pq.h" nogil:

    ctypedef enum codebook_gen:
        PER_SUBSPACE
        PER_CLUSTER

    ctypedef struct cuvsIvfPqIndexParams:
        cuvsDistanceType metric
        float metric_arg
        bool add_data_on_build
        uint32_t n_lists
        uint32_t kmeans_n_iters
        double kmeans_trainset_fraction
        uint32_t pq_bits
        uint32_t pq_dim
        codebook_gen codebook_kind
        bool force_random_rotation
        bool conservative_memory_allocation
        uint32_t max_train_points_per_pq_code

    ctypedef cuvsIvfPqIndexParams* cuvsIvfPqIndexParams_t

    ctypedef struct cuvsIvfPqSearchParams:
        uint32_t n_probes
        cudaDataType_t lut_dtype
        cudaDataType_t internal_distance_dtype
        double preferred_shmem_carveout
        cudaDataType_t coarse_search_dtype
        uint32_t max_internal_batch_size

    ctypedef cuvsIvfPqSearchParams* cuvsIvfPqSearchParams_t

    ctypedef struct cuvsIvfPqIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsIvfPqIndex* cuvsIvfPqIndex_t

    cuvsError_t cuvsIvfPqIndexParamsCreate(cuvsIvfPqIndexParams_t* params)

    cuvsError_t cuvsIvfPqIndexParamsDestroy(cuvsIvfPqIndexParams_t index)

    cuvsError_t cuvsIvfPqSearchParamsCreate(
        cuvsIvfPqSearchParams_t* params)

    cuvsError_t cuvsIvfPqSearchParamsDestroy(cuvsIvfPqSearchParams_t index)

    cuvsError_t cuvsIvfPqIndexCreate(cuvsIvfPqIndex_t* index)

    cuvsError_t cuvsIvfPqIndexDestroy(cuvsIvfPqIndex_t index)

    cuvsError_t cuvsIvfPqIndexGetNLists(cuvsIvfPqIndex_t index,
                                        int64_t * n_lists)

    cuvsError_t cuvsIvfPqIndexGetDim(cuvsIvfPqIndex_t index, int64_t * dim)

    cuvsError_t cuvsIvfPqIndexGetSize(cuvsIvfPqIndex_t index, int64_t * size)

    cuvsError_t cuvsIvfPqIndexGetPqDim(cuvsIvfPqIndex_t index,
                                       int64_t * pq_dim)

    cuvsError_t cuvsIvfPqIndexGetPqBits(cuvsIvfPqIndex_t index,
                                        int64_t * pq_bits)

    cuvsError_t cuvsIvfPqIndexGetPqLen(cuvsIvfPqIndex_t index,
                                       int64_t * pq_len)

    cuvsError_t cuvsIvfPqIndexGetCenters(cuvsIvfPqIndex_t index,
                                         DLManagedTensor * centers)

    cuvsError_t cuvsIvfPqIndexGetCentersPadded(cuvsIvfPqIndex_t index,
                                               DLManagedTensor * centers)

    cuvsError_t cuvsIvfPqIndexGetListSizes(cuvsIvfPqIndex_t index,
                                           DLManagedTensor * list_sizes)

    cuvsError_t cuvsIvfPqIndexGetPqCenters(cuvsIvfPqIndex_t index,
                                           DLManagedTensor * centers)

    cuvsError_t cuvsIvfPqIndexGetCentersRot(cuvsIvfPqIndex_t index,
                                            DLManagedTensor * centers_rot)

    cuvsError_t cuvsIvfPqIndexGetRotationMatrix(cuvsIvfPqIndex_t index,
                                                DLManagedTensor * rotation_matrix)

    cuvsError_t cuvsIvfPqIndexUnpackContiguousListData(cuvsResources_t res,
                                                       cuvsIvfPqIndex_t index,
                                                       DLManagedTensor* out,
                                                       uint32_t label,
                                                       uint32_t offset)

    cuvsError_t cuvsIvfPqIndexGetListIndices(cuvsIvfPqIndex_t index,
                                             uint32_t label,
                                             DLManagedTensor* out)

    cuvsError_t cuvsIvfPqBuild(cuvsResources_t res,
                               cuvsIvfPqIndexParams* params,
                               DLManagedTensor* dataset,
                               cuvsIvfPqIndex_t index)

    cuvsError_t cuvsIvfPqBuildPrecomputed(cuvsResources_t res,
                                          cuvsIvfPqIndexParams_t params,
                                          uint32_t dim,
                                          DLManagedTensor* pq_centers,
                                          DLManagedTensor* centers,
                                          DLManagedTensor* centers_rot,
                                          DLManagedTensor* rotation_matrix,
                                          cuvsIvfPqIndex_t index)

    cuvsError_t cuvsIvfPqSearch(cuvsResources_t res,
                                cuvsIvfPqSearchParams* params,
                                cuvsIvfPqIndex_t index,
                                DLManagedTensor* queries,
                                DLManagedTensor* neighbors,
                                DLManagedTensor* distances)

    cuvsError_t cuvsIvfPqSerialize(cuvsResources_t res,
                                   const char * filename,
                                   cuvsIvfPqIndex_t index)

    cuvsError_t cuvsIvfPqDeserialize(cuvsResources_t res,
                                     const char * filename,
                                     cuvsIvfPqIndex_t index)

    cuvsError_t cuvsIvfPqExtend(cuvsResources_t res,
                                DLManagedTensor* new_vectors,
                                DLManagedTensor* new_indices,
                                cuvsIvfPqIndex_t index)


cdef class IndexParams:
    cdef cuvsIvfPqIndexParams* params
    cdef object _metric

cdef class SearchParams:
    cdef cuvsIvfPqSearchParams* params
