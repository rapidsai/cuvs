/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup gmm_c_params Gaussian mixture hyperparameters
 * @{
 */

/**
 * @brief Covariance parameterization of the mixture components.
 */
typedef enum {
  /** Each component has its own full covariance matrix. */
  CUVS_GMM_COVARIANCE_FULL = 0,
  /** All components share a single full covariance matrix. */
  CUVS_GMM_COVARIANCE_TIED = 1,
  /** Each component has its own diagonal covariance. */
  CUVS_GMM_COVARIANCE_DIAG = 2,
  /** Each component has a single variance. */
  CUVS_GMM_COVARIANCE_SPHERICAL = 3
} cuvsGMMCovarianceType;

/**
 * @brief Strategy used to initialize the responsibilities before EM.
 */
typedef enum {
  /** Run k-means (itself seeded with k-means++) and use the hard labels. */
  CUVS_GMM_INIT_KMEANS = 0,
  /** Use the k-means++ seeding labels directly. */
  CUVS_GMM_INIT_KMEANS_PLUS_PLUS = 1,
  /** Random per-sample-normalized responsibilities. */
  CUVS_GMM_INIT_RANDOM = 2,
  /** Pick n_components samples at random as one-hot responsibilities. */
  CUVS_GMM_INIT_RANDOM_FROM_DATA = 3
} cuvsGMMInitMethod;

/**
 * @brief Hyper-parameters for the Gaussian mixture EM solver
 */
struct cuvsGMMParams {
  /**
   * The number of mixture components. Default: 1.
   */
  int n_components;

  /**
   * Covariance parameterization of the mixture components. Default: FULL.
   */
  cuvsGMMCovarianceType covariance_type;

  /**
   * Convergence threshold on the change of the per-sample average
   * log-likelihood (lower bound). Default: 1e-3.
   */
  double tol;

  /**
   * Non-negative regularization added to the diagonal of covariance.
   * Default: 1e-6.
   */
  double reg_covar;

  /**
   * Maximum number of EM iterations for a single run. Default: 100.
   */
  int max_iter;

  /**
   * Number of initializations to perform; the best result is kept. Default: 1.
   */
  int n_init;

  /**
   * Strategy used to initialize the responsibilities before EM.
   * Default: KMEANS.
   */
  cuvsGMMInitMethod init;

  /**
   * Seed to the random number generator. Default: 0.
   */
  uint64_t seed;
};

typedef struct cuvsGMMParams* cuvsGMMParams_t;

/**
 * @brief Allocate GMM params, and populate with default values
 *
 * @param[in] params cuvsGMMParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsGMMParamsCreate(cuvsGMMParams_t* params);

/**
 * @brief De-allocate GMM params
 *
 * @param[in] params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsGMMParamsDestroy(cuvsGMMParams_t params);

/**
 * @}
 */

/**
 * @defgroup gmm_c Gaussian mixture model APIs
 * @{
 *
 * The covariance-shaped tensors (``covariances``, ``precisions_chol``,
 * ``precisions``) depend on ``covariance_type``. With ``K = n_components``
 * and ``d = n_features`` the expected shapes are (row-major):
 *
 *   - ``CUVS_GMM_COVARIANCE_FULL``:      (K, d, d)
 *   - ``CUVS_GMM_COVARIANCE_TIED``:      (d, d)
 *   - ``CUVS_GMM_COVARIANCE_DIAG``:      (K, d)
 *   - ``CUVS_GMM_COVARIANCE_SPHERICAL``: (K,)
 */

/**
 * @brief Fit a Gaussian mixture with the EM algorithm.
 *
 * Runs ``params->n_init`` random restarts (unless ``warm_start`` is true) and
 * keeps the parameters with the largest lower bound.
 *
 * All tensors must reside on device memory and be row-major. ``X``,
 * ``weights``, ``means``, ``covariances``, ``precisions_chol`` and
 * ``precisions`` must share one dtype (float32 or float64); ``labels`` is
 * int32.
 *
 * @param[in]    res             opaque C handle
 * @param[in]    params          Parameters for the GMM model.
 * @param[in]    X               Training data. [dim = n_samples x n_features]
 * @param[inout] weights         Mixture weights. [len = n_components]
 * @param[inout] means           Component means.
 *                               [dim = n_components x n_features]
 * @param[inout] covariances     Component covariances, flat. Length by
 *                               covariance_type (K=n_components, d=n_features):
 *                               FULL K*d*d, TIED d*d, DIAG K*d, SPHERICAL K.
 * @param[out]   precisions_chol Precision Cholesky factors, same flat layout as
 *                               covariances (FULL/TIED: upper-triangular factor
 *                               U with precision = U @ Uᵀ; DIAG/SPHERICAL:
 *                               reciprocal standard deviations).
 * @param[out]   precisions      Precision matrices, same flat layout as
 *                               covariances.
 * @param[out]   labels          Hard component assignment per sample.
 *                               [len = n_samples]
 * @param[out]   lower_bound     Per-sample average log-likelihood of the best
 *                               fit.
 * @param[out]   n_iter          Number of EM iterations of the best fit.
 * @param[out]   converged       Whether the best fit converged within tol.
 * @param[in]    warm_start      Use the incoming weights/means/covariances as
 *                               the single initialization.
 */
CUVS_EXPORT cuvsError_t cuvsGMMFit(cuvsResources_t res,
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
                                   bool warm_start);

/**
 * @brief Hard component labels (argmax responsibility) for new data.
 *
 * @param[in]  res             opaque C handle
 * @param[in]  params          Parameters used to fit the GMM model.
 * @param[in]  X               Data to assign. [dim = n_samples x n_features]
 * @param[in]  weights         Fitted mixture weights. [len = n_components]
 * @param[in]  means           Fitted component means.
 *                             [dim = n_components x n_features]
 * @param[in]  precisions_chol Fitted precision Cholesky factors, flat. Length
 *                             by covariance_type (K=n_components, d=n_features):
 *                             FULL K*d*d, TIED d*d, DIAG K*d, SPHERICAL K.
 * @param[out] labels          Hard component assignment per sample (int32).
 *                             [len = n_samples]
 */
CUVS_EXPORT cuvsError_t cuvsGMMPredict(cuvsResources_t res,
                                       cuvsGMMParams_t params,
                                       DLManagedTensor* X,
                                       DLManagedTensor* weights,
                                       DLManagedTensor* means,
                                       DLManagedTensor* precisions_chol,
                                       DLManagedTensor* labels);

/**
 * @brief Posterior responsibilities for new data.
 *
 * @param[in]  res             opaque C handle
 * @param[in]  params          Parameters used to fit the GMM model.
 * @param[in]  X               Data to evaluate. [dim = n_samples x n_features]
 * @param[in]  weights         Fitted mixture weights. [len = n_components]
 * @param[in]  means           Fitted component means.
 *                             [dim = n_components x n_features]
 * @param[in]  precisions_chol Fitted precision Cholesky factors, flat. Length
 *                             by covariance_type (K=n_components, d=n_features):
 *                             FULL K*d*d, TIED d*d, DIAG K*d, SPHERICAL K.
 * @param[out] resp            Posterior probability of each component for
 *                             each sample. [dim = n_samples x n_components]
 */
CUVS_EXPORT cuvsError_t cuvsGMMPredictProba(cuvsResources_t res,
                                            cuvsGMMParams_t params,
                                            DLManagedTensor* X,
                                            DLManagedTensor* weights,
                                            DLManagedTensor* means,
                                            DLManagedTensor* precisions_chol,
                                            DLManagedTensor* resp);

/**
 * @brief Per-sample log-likelihood log p(x_i) for new data.
 *
 * @param[in]  res             opaque C handle
 * @param[in]  params          Parameters used to fit the GMM model.
 * @param[in]  X               Data to evaluate. [dim = n_samples x n_features]
 * @param[in]  weights         Fitted mixture weights. [len = n_components]
 * @param[in]  means           Fitted component means.
 *                             [dim = n_components x n_features]
 * @param[in]  precisions_chol Fitted precision Cholesky factors, flat. Length
 *                             by covariance_type (K=n_components, d=n_features):
 *                             FULL K*d*d, TIED d*d, DIAG K*d, SPHERICAL K.
 * @param[out] log_prob_norm   Log-likelihood of each sample under the model.
 *                             [len = n_samples]
 */
CUVS_EXPORT cuvsError_t cuvsGMMScoreSamples(cuvsResources_t res,
                                            cuvsGMMParams_t params,
                                            DLManagedTensor* X,
                                            DLManagedTensor* weights,
                                            DLManagedTensor* means,
                                            DLManagedTensor* precisions_chol,
                                            DLManagedTensor* log_prob_norm);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
