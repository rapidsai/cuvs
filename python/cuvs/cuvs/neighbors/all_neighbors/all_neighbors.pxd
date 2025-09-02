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

from libc.stdint cimport int64_t

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType
from cuvs.neighbors.ivf_pq.ivf_pq cimport cuvsIvfPqIndexParams_t
from cuvs.neighbors.nn_descent.nn_descent cimport cuvsNNDescentIndexParams_t


cdef extern from "cuvs/neighbors/all_neighbors.h" nogil:
    ctypedef enum cuvsAllNeighborsAlgo:
        CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE
        CUVS_ALL_NEIGHBORS_ALGO_IVF_PQ
        CUVS_ALL_NEIGHBORS_ALGO_NN_DESCENT

    ctypedef struct cuvsAllNeighborsIndexParams:
        cuvsAllNeighborsAlgo algo
        size_t overlap_factor
        size_t n_clusters
        cuvsDistanceType metric

        cuvsIvfPqIndexParams_t ivf_pq_params
        cuvsNNDescentIndexParams_t nn_descent_params

    ctypedef cuvsAllNeighborsIndexParams* cuvsAllNeighborsIndexParams_t

    cuvsError_t cuvsAllNeighborsBuild(
        cuvsResources_t res,
        cuvsAllNeighborsIndexParams_t params,
        DLManagedTensor* dataset,
        DLManagedTensor* indices,
        DLManagedTensor* distances,
        DLManagedTensor* core_distances,
        float alpha
    )
