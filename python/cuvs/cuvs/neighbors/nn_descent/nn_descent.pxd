#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport uint32_t, uintptr_t
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType


cdef extern from "cuvs/neighbors/nn_descent.h" nogil:
    enum cuvsNNDescentDistCompDtype:
        NND_DIST_COMP_AUTO = 0,
        NND_DIST_COMP_FP32 = 1,
        NND_DIST_COMP_FP16 = 2

    ctypedef struct cuvsNNDescentIndexParams:
        cuvsDistanceType metric
        float metric_arg
        size_t graph_degree
        size_t intermediate_graph_degree
        size_t max_iterations
        float termination_threshold
        bool return_distances
        cuvsNNDescentDistCompDtype dist_comp_dtype

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
