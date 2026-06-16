#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport int64_t, uint64_t
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLManagedTensor


cdef extern from "cuvs/cluster/gmm.h" nogil:
    ctypedef enum cuvsGMMCovarianceType:
        CUVS_GMM_COVARIANCE_FULL
        CUVS_GMM_COVARIANCE_TIED
        CUVS_GMM_COVARIANCE_DIAG
        CUVS_GMM_COVARIANCE_SPHERICAL

    ctypedef enum cuvsGMMInitMethod:
        CUVS_GMM_INIT_KMEANS
        CUVS_GMM_INIT_KMEANS_PLUS_PLUS
        CUVS_GMM_INIT_RANDOM
        CUVS_GMM_INIT_RANDOM_FROM_DATA

    ctypedef struct cuvsGMMParams:
        int n_components,
        cuvsGMMCovarianceType covariance_type,
        double tol,
        double reg_covar,
        int max_iter,
        int n_init,
        cuvsGMMInitMethod init,
        uint64_t seed

    ctypedef cuvsGMMParams* cuvsGMMParams_t

    cuvsError_t cuvsGMMParamsCreate(cuvsGMMParams_t* params)

    cuvsError_t cuvsGMMParamsDestroy(cuvsGMMParams_t params)

    cuvsError_t cuvsGMMFit(cuvsResources_t res,
                           cuvsGMMParams_t params,
                           DLManagedTensor* X,
                           DLManagedTensor* weights,
                           DLManagedTensor* means,
                           DLManagedTensor* covariances,
                           DLManagedTensor* precisions_chol,
                           DLManagedTensor* precisions,
                           DLManagedTensor* labels,
                           double* lower_bound,
                           int* n_iter,
                           bool* converged,
                           bool warm_start) except +

    cuvsError_t cuvsGMMPredict(cuvsResources_t res,
                               cuvsGMMParams_t params,
                               DLManagedTensor* X,
                               DLManagedTensor* weights,
                               DLManagedTensor* means,
                               DLManagedTensor* precisions_chol,
                               DLManagedTensor* labels) except +

    cuvsError_t cuvsGMMPredictProba(cuvsResources_t res,
                                    cuvsGMMParams_t params,
                                    DLManagedTensor* X,
                                    DLManagedTensor* weights,
                                    DLManagedTensor* means,
                                    DLManagedTensor* precisions_chol,
                                    DLManagedTensor* resp) except +

    cuvsError_t cuvsGMMScoreSamples(cuvsResources_t res,
                                    cuvsGMMParams_t params,
                                    DLManagedTensor* X,
                                    DLManagedTensor* weights,
                                    DLManagedTensor* means,
                                    DLManagedTensor* precisions_chol,
                                    DLManagedTensor* log_prob_norm) except +
