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

from libc.stdint cimport int32_t, uintptr_t

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType


cdef extern from "cuvs/neighbors/hnsw.h" nogil:
    ctypedef struct cuvsHnswSearchParams:
        int32_t ef
        int32_t numThreads

    ctypedef cuvsHnswSearchParams* cuvsHnswSearchParams_t

    ctypedef struct cuvsHnswIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsHnswIndex* cuvsHnswIndex_t

    cuvsError_t cuvsHnswIndexCreate(cuvsHnswIndex_t* index)

    cuvsError_t cuvsHnswIndexDestroy(cuvsHnswIndex_t index)

    cuvsError_t cuvsHnswSearch(cuvsResources_t res,
                               cuvsHnswSearchParams* params,
                               cuvsHnswIndex_t index,
                               DLManagedTensor* queries,
                               DLManagedTensor* neighbors,
                               DLManagedTensor* distances) except +

    cuvsError_t cuvsHnswDeserialize(cuvsResources_t res,
                                    const char * filename,
                                    int32_t dim,
                                    cuvsDistanceType metric,
                                    cuvsHnswIndex_t index) except +
