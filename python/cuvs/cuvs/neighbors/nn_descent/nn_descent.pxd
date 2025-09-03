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

from libc.stdint cimport uint32_t, uintptr_t
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType


cdef extern from "cuvs/neighbors/nn_descent.h" nogil:

    ctypedef struct cuvsNNDescentIndexParams:
        cuvsDistanceType metric
        float metric_arg
        size_t graph_degree
        size_t intermediate_graph_degree
        size_t max_iterations
        float termination_threshold
        bool return_distances

    ctypedef cuvsNNDescentIndexParams* cuvsNNDescentIndexParams_t

    ctypedef struct cuvsNNDescentIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsNNDescentIndex* cuvsNNDescentIndex_t

    cuvsError_t cuvsNNDescentIndexParamsCreate(
        cuvsNNDescentIndexParams_t* params)

    cuvsError_t cuvsNNDescentIndexParamsDestroy(
        cuvsNNDescentIndexParams_t index)

    cuvsError_t cuvsNNDescentIndexCreate(cuvsNNDescentIndex_t* index)

    cuvsError_t cuvsNNDescentIndexDestroy(cuvsNNDescentIndex_t index)

    cuvsError_t cuvsNNDescentIndexGetGraph(cuvsResources_t res,
                                           cuvsNNDescentIndex_t index,
                                           DLManagedTensor * output)

    cuvsError_t cuvsNNDescentIndexGetDistances(cuvsResources_t res,
                                               cuvsNNDescentIndex_t index,
                                               DLManagedTensor * output)

    cuvsError_t cuvsNNDescentBuild(cuvsResources_t res,
                                   cuvsNNDescentIndexParams* params,
                                   DLManagedTensor* dataset,
                                   DLManagedTensor* graph,
                                   cuvsNNDescentIndex_t index) except +
