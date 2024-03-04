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

from libc.stdint cimport (
    int8_t,
    int64_t,
    uint8_t,
    uint32_t,
    uint64_t,
    uintptr_t,
)

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor


cdef extern from "cuvs/neighbors/cagra.h" nogil:

    ctypedef enum cuvsCagraGraphBuildAlgo:
        IVF_PQ
        NN_DESCENT

    ctypedef struct cuvsCagraIndexParams:
        size_t intermediate_graph_degree
        size_t graph_degree
        cuvsCagraGraphBuildAlgo build_algo
        size_t nn_descent_niter

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

    ctypedef struct cuvsCagraIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsCagraIndex* cuvsCagraIndex_t

    cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t* params)

    cuvsError_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t index)

    cuvsError_t cuvsCagraIndexCreate(cuvsCagraIndex_t* index)

    cuvsError_t cuvsCagraIndexDestroy(cuvsCagraIndex_t index)

    cuvsError_t cuvsCagraBuild(cuvsResources_t res,
                               cuvsCagraIndexParams* params,
                               DLManagedTensor* dataset,
                               cuvsCagraIndex_t index) except +

    cuvsError_t cuvsCagraSearch(cuvsResources_t res,
                                cuvsCagraSearchParams* params,
                                cuvsCagraIndex_t index,
                                DLManagedTensor* queries,
                                DLManagedTensor* neighbors,
                                DLManagedTensor* distances) except +
