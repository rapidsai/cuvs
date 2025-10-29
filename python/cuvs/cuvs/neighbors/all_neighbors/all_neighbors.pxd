#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
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
