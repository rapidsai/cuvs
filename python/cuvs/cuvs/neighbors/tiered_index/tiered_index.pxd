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

from libc.stdint cimport int64_t, uint32_t, uintptr_t
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType
from cuvs.neighbors.cagra.cagra cimport cuvsCagraIndexParams_t
from cuvs.neighbors.filters.filters cimport cuvsFilter
from cuvs.neighbors.ivf_flat.ivf_flat cimport cuvsIvfFlatIndexParams_t
from cuvs.neighbors.ivf_pq.ivf_pq cimport cuvsIvfPqIndexParams_t


cdef extern from "cuvs/neighbors/tiered_index.h" nogil:
    ctypedef enum cuvsTieredIndexANNAlgo:
        CUVS_TIERED_INDEX_ALGO_CAGRA
        CUVS_TIERED_INDEX_ALGO_IVF_FLAT
        CUVS_TIERED_INDEX_ALGO_IVF_PQ

    ctypedef struct cuvsTieredIndexParams:
        cuvsDistanceType metric
        cuvsTieredIndexANNAlgo algo
        int64_t min_ann_rows
        bool create_ann_index_on_extend

        cuvsCagraIndexParams_t cagra_params
        cuvsIvfFlatIndexParams_t ivf_flat_params
        cuvsIvfPqIndexParams_t ivf_pq_params

    ctypedef cuvsTieredIndexParams* cuvsTieredIndexParams_t

    ctypedef struct cuvsTieredIndex:
        uintptr_t addr
        DLDataType dtype
        cuvsTieredIndexANNAlgo algo

    ctypedef cuvsTieredIndex* cuvsTieredIndex_t

    cuvsError_t cuvsTieredIndexParamsCreate(cuvsTieredIndexParams_t* params)

    cuvsError_t cuvsTieredIndexParamsDestroy(cuvsTieredIndexParams_t index)

    cuvsError_t cuvsTieredIndexCreate(cuvsTieredIndex_t* index)

    cuvsError_t cuvsTieredIndexDestroy(cuvsTieredIndex_t index)

    cuvsError_t cuvsTieredIndexBuild(cuvsResources_t res,
                                     cuvsTieredIndexParams* params,
                                     DLManagedTensor* dataset,
                                     cuvsTieredIndex_t index)

    cuvsError_t cuvsTieredIndexSearch(cuvsResources_t res,
                                      void * params,
                                      cuvsTieredIndex_t index,
                                      DLManagedTensor* queries,
                                      DLManagedTensor* neighbors,
                                      DLManagedTensor* distances,
                                      cuvsFilter filter) except +

    cuvsError_t cuvsTieredIndexExtend(cuvsResources_t res,
                                      DLManagedTensor* new_vectors,
                                      cuvsTieredIndex_t index)
