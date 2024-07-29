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


cdef extern from "cuvs/neighbors/ivf_flat.h" nogil:

    ctypedef struct cuvsIvfFlatIndexParams:
        cuvsDistanceType metric
        float metric_arg
        bool add_data_on_build
        uint32_t n_lists
        uint32_t kmeans_n_iters
        double kmeans_trainset_fraction
        bool adaptive_centers
        bool conservative_memory_allocation

    ctypedef cuvsIvfFlatIndexParams* cuvsIvfFlatIndexParams_t

    ctypedef struct cuvsIvfFlatSearchParams:
        uint32_t n_probes

    ctypedef cuvsIvfFlatSearchParams* cuvsIvfFlatSearchParams_t

    ctypedef struct cuvsIvfFlatIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsIvfFlatIndex* cuvsIvfFlatIndex_t

    cuvsError_t cuvsIvfFlatIndexParamsCreate(cuvsIvfFlatIndexParams_t* params)

    cuvsError_t cuvsIvfFlatIndexParamsDestroy(cuvsIvfFlatIndexParams_t index)

    cuvsError_t cuvsIvfFlatSearchParamsCreate(
        cuvsIvfFlatSearchParams_t* params)

    cuvsError_t cuvsIvfFlatSearchParamsDestroy(cuvsIvfFlatSearchParams_t index)

    cuvsError_t cuvsIvfFlatIndexCreate(cuvsIvfFlatIndex_t* index)

    cuvsError_t cuvsIvfFlatIndexDestroy(cuvsIvfFlatIndex_t index)

    cuvsError_t cuvsIvfFlatBuild(cuvsResources_t res,
                                 cuvsIvfFlatIndexParams* params,
                                 DLManagedTensor* dataset,
                                 cuvsIvfFlatIndex_t index) except +

    cuvsError_t cuvsIvfFlatSearch(cuvsResources_t res,
                                  cuvsIvfFlatSearchParams* params,
                                  cuvsIvfFlatIndex_t index,
                                  DLManagedTensor* queries,
                                  DLManagedTensor* neighbors,
                                  DLManagedTensor* distances) except +

    cuvsError_t cuvsIvfFlatSerialize(cuvsResources_t res,
                                     const char * filename,
                                     cuvsIvfFlatIndex_t index) except +

    cuvsError_t cuvsIvfFlatDeserialize(cuvsResources_t res,
                                       const char * filename,
                                       cuvsIvfFlatIndex_t index) except +

    cuvsError_t cuvsIvfFlatExtend(cuvsResources_t res,
                                  DLManagedTensor* new_vectors,
                                  DLManagedTensor* new_indices,
                                  cuvsIvfFlatIndex_t index)
