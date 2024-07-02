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

from libc.stdint cimport uintptr_t

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType
from cuvs.neighbors.filters.filters cimport cuvsFilter, cuvsFilterType


cdef extern from "cuvs/neighbors/brute_force.h" nogil:

    ctypedef struct cuvsBruteForceIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsBruteForceIndex* cuvsBruteForceIndex_t

    cuvsError_t cuvsBruteForceIndexCreate(cuvsBruteForceIndex_t* index)

    cuvsError_t cuvsBruteForceIndexDestroy(cuvsBruteForceIndex_t index)

    cuvsError_t cuvsBruteForceBuild(cuvsResources_t res,
                                    DLManagedTensor* dataset,
                                    cuvsDistanceType metric,
                                    float metric_arg,
                                    cuvsBruteForceIndex_t index) except +

    cuvsError_t cuvsBruteForceSearch(cuvsResources_t res,
                                     cuvsBruteForceIndex_t index,
                                     DLManagedTensor* queries,
                                     DLManagedTensor* neighbors,
                                     DLManagedTensor* distances,
                                     cuvsFilter filter) except +
