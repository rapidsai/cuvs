/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

enum cuvsKMeansInitMethod {
  /**
   * Sample the centroids using the kmeans++ strategy
   */
  KMeansPlusPlus,

  /**
   * Sample the centroids uniformly at random
   */
  Random,

  /**
   * User provides the array of initial centroids
   */
  Array
};

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

  bool inertia_check;

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
 * @param[in]     res           opaque C handle
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional weights for each observation in X.
 *                              [len = n_samples]
 * @param[inout]  centroids     [in] When init is InitMethod::Array, use
 *                              centroids as the initial cluster centers.
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
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
 * @}
 */

#ifdef __cplusplus
}
#endif
