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

from libc.stdint cimport int8_t, int64_t, uint8_t, uint32_t, uint64_t, uintptr_t

from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t


cdef extern from "cuvs/neighbors/cagra_c.h" nogil:

    ctypedef enum cagraGraphBuildAlgo:
        IVF_PQ
        NN_DESCENT


    ctypedef struct cagraIndexParams:
        size_t intermediate_graph_degree
        size_t graph_degree
        cagraGraphBuildAlgo build_algo
        size_t nn_descent_niter


    ctypedef enum cagraSearchAlgo:
        SINGLE_CTA,
        MULTI_CTA,
        MULTI_KERNEL,
        AUTO

    ctypedef enum cagraHashMode:
        HASH,
        SMALL,
        AUTO_HASH

    ctypedef struct cagraSearchParams:
        size_t max_queries
        size_t itopk_size
        size_t max_iterations
        cagraSearchAlgo algo
        size_t team_size
        size_t search_width
        size_t min_iterations
        size_t thread_block_size
        cagraHashMode hashmap_mode
        size_t hashmap_min_bitlen
        float hashmap_max_fill_rate
        uint32_t num_random_samplings
        uint64_t rand_xor_mask

    ctypedef struct cagraIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cagraIndex* cagraIndex_t

    cuvsError_t cagraIndexCreate(cagraIndex_t* index)

    cuvsError_t cagraIndexDestroy(cagraIndex_t index)

    cuvsError_t cagraBuild(cuvsResources_t res,
                           cagraIndexParams params,
                           DLManagedTensor* dataset,
                           cagraIndex_t index);

    cuvsError_t cagraSearch(cuvsResources_t res,
                            cagraSearchParams params,
                            cagraIndex_t index,
                            DLManagedTensor* queries,
                            DLManagedTensor* neighbors,
                            DLManagedTensor* distances)
