#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType


cdef extern from "cuvs/cluster/kmeans.h" nogil:
    ctypedef enum cuvsKMeansInitMethod:
        KMeansPlusPlus
        Random
        Array

    ctypedef enum cuvsKMeansCentroidUpdateMode:
        CUVS_KMEANS_UPDATE_FULL_BATCH
        CUVS_KMEANS_UPDATE_MINI_BATCH

    ctypedef enum cuvsKMeansType:
        CUVS_KMEANS_TYPE_KMEANS
        CUVS_KMEANS_TYPE_KMEANS_BALANCED

    ctypedef struct cuvsKMeansParams:
        cuvsDistanceType metric,
        int n_clusters,
        cuvsKMeansInitMethod init,
        int max_iter,
        double tol,
        int n_init,
        double oversampling_factor,
        int batch_samples,
        int batch_centroids,
        cuvsKMeansCentroidUpdateMode update_mode,
        bool inertia_check,
        bool final_inertia_check,
        int max_no_improvement,
        double reassignment_ratio,
        bool hierarchical,
        int hierarchical_n_iters

    ctypedef cuvsKMeansParams* cuvsKMeansParams_t

    cuvsError_t cuvsKMeansParamsCreate(cuvsKMeansParams_t* index)

    cuvsError_t cuvsKMeansParamsDestroy(cuvsKMeansParams_t index)

    cuvsError_t cuvsKMeansFit(cuvsResources_t res,
                              cuvsKMeansParams_t params,
                              DLManagedTensor* X,
                              DLManagedTensor* sample_weight,
                              DLManagedTensor * centroids,
                              double * inertia,
                              int * n_iter) except +

    cuvsError_t cuvsKMeansPredict(cuvsResources_t res,
                                  cuvsKMeansParams_t params,
                                  DLManagedTensor* X,
                                  DLManagedTensor* sample_weight,
                                  DLManagedTensor * centroids,
                                  DLManagedTensor * labels,
                                  bool normalize_weight,
                                  double * inertia)

    cuvsError_t cuvsKMeansClusterCost(cuvsResources_t res,
                                      DLManagedTensor* X,
                                      DLManagedTensor* centroids,
                                      double* cost)

    cuvsError_t cuvsKMeansFitBatched(cuvsResources_t res,
                                     cuvsKMeansParams_t params,
                                     DLManagedTensor* X,
                                     int64_t batch_size,
                                     DLManagedTensor* sample_weight,
                                     DLManagedTensor* centroids,
                                     double* inertia,
                                     int64_t* n_iter) except +
