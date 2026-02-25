/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup kmeans_c_params k-means hyperparameters
 * @{
 */

typedef enum {
  /**
   * Sample the centroids using the kmeans++ strategy
   */
  KMeansPlusPlus = 0,

  /**
   * Sample the centroids uniformly at random
   */
  Random = 1,

  /**
   * User provides the array of initial centroids
   */
  Array = 2
} cuvsKMeansInitMethod;

/**
 * @brief Hyper-parameters for the kmeans algorithm
 */
struct cuvsKMeansParams {
  cuvsDistanceType metric;

  /**
   * The number of clusters to form as well as the number of centroids to generate (default:8).
   */
  int n_clusters;

  /**
   * Method for initialization, defaults to k-means++:
   *  - cuvsKMeansInitMethod::KMeansPlusPlus (k-means++): Use scalable k-means++ algorithm
   * to select the initial cluster centers.
   *  - cuvsKMeansInitMethod::Random (random): Choose 'n_clusters' observations (rows) at
   * random from the input data for the initial centroids.
   *  - cuvsKMeansInitMethod::Array (ndarray): Use 'centroids' as initial cluster centers.
   */
  cuvsKMeansInitMethod init;

  /**
   * Maximum number of iterations of the k-means algorithm for a single run.
   */
  int max_iter;

  /**
   * Relative tolerance with regards to inertia to declare convergence.
   */
  double tol;

  /**
   * Number of instance k-means algorithm will be run with different seeds.
   */
  int n_init;

  /**
   * Oversampling factor for use in the k-means|| algorithm
   */
  double oversampling_factor;

  /**
   * batch_samples and batch_centroids are used to tile 1NN computation which is
   * useful to optimize/control the memory footprint
   * Default tile is [batch_samples x n_clusters] i.e. when batch_centroids is 0
   * then don't tile the centroids
   */
  int batch_samples;

  /**
   * if 0 then batch_centroids = n_clusters
   */
  int batch_centroids;

  /** Check inertia during iterations for early convergence. */
  bool inertia_check;

  /**
   * Number of samples to process per batch. Controls memory usage when the
   * dataset is large or resides on the host. If 0 (default), the entire dataset
   * is used as a single batch.
   *
   * For cuvsKMeansFit with host data, this controls the streaming batch size.
   * For cuvsMiniBatchKMeansFit, this is the mini-batch size.
   */
  int batch_size;

  /**
   * Compute final inertia after fit completes (requires extra data pass).
   * Only relevant when data is on host; regular device-data fit always computes final inertia.
   */
  bool final_inertia_check;

  /**
   * Maximum number of consecutive mini-batch steps without improvement in smoothed inertia
   * before early stopping. Only used by cuvsMiniBatchKMeansFit.
   * If 0, this convergence criterion is disabled.
   */
  int max_no_improvement;

  /**
   * Control the fraction of the maximum number of counts for a center to be reassigned.
   * Centers with count < reassignment_ratio * max(counts) are randomly reassigned to
   * observations from the current batch. Only used by cuvsMiniBatchKMeansFit.
   * If 0.0, reassignment is disabled. Default: 0.01
   */
  double reassignment_ratio;

  /**
   * Whether to use hierarchical (balanced) kmeans or not
   */
  bool hierarchical;

  /**
   * For hierarchical k-means , defines the number of training iterations
   */
  int hierarchical_n_iters;
};

typedef struct cuvsKMeansParams* cuvsKMeansParams_t;

/**
 * @brief Allocate KMeans params, and populate with default values
 *
 * @param[in] params cuvsKMeansParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsKMeansParamsCreate(cuvsKMeansParams_t* params);

/**
 * @brief De-allocate KMeans params
 *
 * @param[in] params
 * @return cuvsError_t
 */
cuvsError_t cuvsKMeansParamsDestroy(cuvsKMeansParams_t params);

/**
 * @brief Type of k-means algorithm.
 */
typedef enum { CUVS_KMEANS_TYPE_KMEANS = 0, CUVS_KMEANS_TYPE_KMEANS_BALANCED = 1 } cuvsKMeansType;

/**
 * @}
 */

/**
 * @defgroup kmeans_c k-means clustering APIs
 * @{
 */

/**
 * @brief Find clusters with k-means algorithm.
 *
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 *
 *   If X is on device (GPU) memory, a standard device-data fit is performed.
 *   If X is on host (CPU) memory, data is automatically streamed to the GPU
 *   in batches controlled by params->batch_size.
 *
 * @param[in]     res           opaque C handle
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster (host or device memory).
 *                              The data must be in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional weights for each observation in X.
 *                              Must be on the same memory space as X.
 *                              [len = n_samples]
 * @param[inout]  centroids     [in] When init is InitMethod::Array, use
 *                              centroids as the initial cluster centers.
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'. Must be on device memory.
 *                              [dim = n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
cuvsError_t cuvsKMeansFit(cuvsResources_t res,
                          cuvsKMeansParams_t params,
                          DLManagedTensor* X,
                          DLManagedTensor* sample_weight,
                          DLManagedTensor* centroids,
                          double* inertia,
                          int* n_iter);

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @param[in]     res              opaque C handle
 * @param[in]     params           Parameters for KMeans model.
 * @param[in]     X                New data to predict.
 *                                 [dim = n_samples x n_features]
 * @param[in]     sample_weight    Optional weights for each observation in X.
 *                                 [len = n_samples]
 * @param[in]     centroids        Cluster centroids. The data must be in
 *                                 row-major format.
 *                                 [dim = n_clusters x n_features]
 * @param[in]     normalize_weight True if the weights should be normalized
 * @param[out]    labels           Index of the cluster each sample in X
 *                                 belongs to.
 *                                 [len = n_samples]
 * @param[out]    inertia          Sum of squared distances of samples to
 *                                 their closest cluster center.
 */
cuvsError_t cuvsKMeansPredict(cuvsResources_t res,
                              cuvsKMeansParams_t params,
                              DLManagedTensor* X,
                              DLManagedTensor* sample_weight,
                              DLManagedTensor* centroids,
                              DLManagedTensor* labels,
                              bool normalize_weight,
                              double* inertia);

/**
 * @brief Compute cluster cost
 *
 * @param[in]  res            opaque C handle
 * @param[in]  X              Training instances to cluster. The data must
 *                            be in row-major format.
 *                            [dim = n_samples x n_features]
 * @param[in]  centroids      Cluster centroids. The data must be in
 *                            row-major format.
 *                            [dim = n_clusters x n_features]
 * @param[out] cost           Resulting cluster cost
 *
 */
cuvsError_t cuvsKMeansClusterCost(cuvsResources_t res,
                                  DLManagedTensor* X,
                                  DLManagedTensor* centroids,
                                  double* cost);

/**
 * @brief Fit mini-batch k-means using host-resident data.
 *
 *   Mini-batches are randomly sampled from the host data each step. Centroids
 *   are updated using an online learning rule. The mini-batch size is
 *   controlled by params->batch_size.
 *
 *   When sample weights are provided, they are used as sampling probabilities
 *   (matching scikit-learn). Unit weights are passed to the centroid update
 *   to avoid double weighting.
 *
 * @param[in]     res           opaque C handle
 * @param[in]     params        Parameters for KMeans model. The fields
 *                              batch_size, max_no_improvement, and
 *                              reassignment_ratio are used by this function.
 * @param[in]     X             Training instances on HOST memory. The data must
 *                              be in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional weights for each observation in X (on host).
 *                              [len = n_samples]
 * @param[inout]  centroids     [in] Initial centroids / [out] Fitted centroids.
 *                              Must be on DEVICE memory.
 *                              [dim = n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of mini-batch steps run.
 */
cuvsError_t cuvsMiniBatchKMeansFit(cuvsResources_t res,
                                   cuvsKMeansParams_t params,
                                   DLManagedTensor* X,
                                   DLManagedTensor* sample_weight,
                                   DLManagedTensor* centroids,
                                   double* inertia,
                                   int64_t* n_iter);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
