/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/cluster/kmeans.h>
#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup mg_kmeans_c Multi-GPU k-means clustering APIs
 * @{
 */

/**
 * @brief Find clusters with single-node multi-GPU k-means using host data.
 *
 * X, sample_weight, and centroids must be host-accessible, row-major,
 * C-contiguous DLPack tensors. X and centroids must have dtype float32 or
 * float64, and sample_weight must match X when provided.
 *
 * @note In cuVS 26.08 (next ABI major version) this signature will be
 * replaced by cuvsMultiGpuKMeansFit_v2.
 *
 * @param[in]     res           cuvsMultiGpuResources_t opaque C handle
 *                              created by cuvsMultiGpuResourcesCreate or
 *                              cuvsMultiGpuResourcesCreateWithDeviceIds.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Host training instances to cluster.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional host weights for each observation in X.
 *                              [len = n_samples]
 * @param[inout]  centroids     Host centroids. When init is Array, used as the
 *                              initial cluster centers. The final generated
 *                              centroids are copied back to this tensor.
 *                              [dim = n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
CUVS_EXPORT cuvsError_t cuvsMultiGpuKMeansFit(cuvsResources_t res,
                                              cuvsKMeansParams_t params,
                                              DLManagedTensor* X,
                                              DLManagedTensor* sample_weight,
                                              DLManagedTensor* centroids,
                                              double* inertia,
                                              int* n_iter);

/**
 * @brief Find clusters with single-node multi-GPU k-means (v2 params layout).
 *
 * Mirrors cuvsMultiGpuKMeansFit but takes cuvsKMeansParams_v2_t. Will become
 * the unsuffixed cuvsMultiGpuKMeansFit in cuVS 26.08.
 *
 * @param[in]     res           cuvsMultiGpuResources_t opaque C handle.
 * @param[in]     params        Parameters for KMeans model (v2 layout).
 * @param[in]     X             Host training instances to cluster.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional host weights for each observation in X.
 *                              [len = n_samples]
 * @param[inout]  centroids     Host centroids. When init is Array, used as the
 *                              initial cluster centers. The final generated
 *                              centroids are copied back to this tensor.
 *                              [dim = n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
CUVS_EXPORT cuvsError_t cuvsMultiGpuKMeansFit_v2(cuvsResources_t res,
                                                 cuvsKMeansParams_v2_t params,
                                                 DLManagedTensor* X,
                                                 DLManagedTensor* sample_weight,
                                                 DLManagedTensor* centroids,
                                                 double* inertia,
                                                 int* n_iter);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
