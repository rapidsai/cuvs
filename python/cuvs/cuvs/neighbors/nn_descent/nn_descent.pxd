#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport uint32_t, uintptr_t
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType


cdef extern from "cuvs/neighbors/nn_descent.h" nogil:
    # Deprecated — to be removed in 26.08 and replaced by cuvsNNDescentIndexParams_v6.
    ctypedef enum cuvsNNDescentDistCompDtype:
        NND_DIST_COMP_AUTO
        NND_DIST_COMP_FP32
        NND_DIST_COMP_FP16

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

    ctypedef struct cuvsNNDescentIndexParams_v6:
        cuvsDistanceType metric
        float metric_arg
        size_t graph_degree
        size_t intermediate_graph_degree
        size_t max_iterations
        float termination_threshold
        bool return_distances
        bool use_fp16_dist_comp

    ctypedef cuvsNNDescentIndexParams_v6* cuvsNNDescentIndexParams_v6_t

    ctypedef struct cuvsNNDescentIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsNNDescentIndex* cuvsNNDescentIndex_t

    cuvsError_t cuvsNNDescentIndexParamsCreate_v6(
        cuvsNNDescentIndexParams_v6_t* params)

    cuvsError_t cuvsNNDescentIndexParamsDestroy_v6(
        cuvsNNDescentIndexParams_v6_t index)

    cuvsError_t cuvsNNDescentIndexCreate(cuvsNNDescentIndex_t* index)

    cuvsError_t cuvsNNDescentIndexDestroy(cuvsNNDescentIndex_t index)

    cuvsError_t cuvsNNDescentIndexGetGraph(cuvsResources_t res,
                                           cuvsNNDescentIndex_t index,
                                           DLManagedTensor * output)

    cuvsError_t cuvsNNDescentIndexGetDistances(cuvsResources_t res,
                                               cuvsNNDescentIndex_t index,
                                               DLManagedTensor * output)

    cuvsError_t cuvsNNDescentBuild_v6(cuvsResources_t res,
                                      cuvsNNDescentIndexParams_v6* params,
                                      DLManagedTensor* dataset,
                                      DLManagedTensor* graph,
                                      cuvsNNDescentIndex_t index) except +
