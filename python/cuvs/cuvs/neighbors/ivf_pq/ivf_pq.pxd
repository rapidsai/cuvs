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

from libc.stdint cimport uint32_t, uintptr_t
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

    cuvsError_t cuvsIvfPqBuild(cuvsResources_t res,
                               cuvsIvfPqIndexParams* params,
                               DLManagedTensor* dataset,
                               cuvsIvfPqIndex_t index) except +

    cuvsError_t cuvsIvfPqSearch(cuvsResources_t res,
                                cuvsIvfPqSearchParams* params,
                                cuvsIvfPqIndex_t index,
                                DLManagedTensor* queries,
                                DLManagedTensor* neighbors,
                                DLManagedTensor* distances) except +

    cuvsError_t cuvsIvfPqSerialize(cuvsResources_t res,
                                   const char * filename,
                                   cuvsIvfPqIndex_t index) except +

    cuvsError_t cuvsIvfPqDeserialize(cuvsResources_t res,
                                     const char * filename,
                                     cuvsIvfPqIndex_t index) except +

    cuvsError_t cuvsIvfPqExtend(cuvsResources_t res,
                                DLManagedTensor* new_vectors,
                                DLManagedTensor* new_indices,
                                cuvsIvfPqIndex_t index)
