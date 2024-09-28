/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng_state.hpp>

namespace cuvs::cluster::kmeans {

/** Base structure for parameters that are common to all k-means algorithms */
struct base_params {
  /**
   * Metric to use for distance computation. The supported metrics can vary per algorithm.
   */
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
};

/**
 * @defgroup kmeans_params k-means hyperparameters
 * @{
 */

/**
 * Simple object to specify hyper-parameters to the kmeans algorithm.
 */
struct params : base_params {
  enum InitMethod {

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
   * The number of clusters to form as well as the number of centroids to generate (default:8).
   */
  int n_clusters = 8;

  /**
   * Method for initialization, defaults to k-means++:
   *  - InitMethod::KMeansPlusPlus (k-means++): Use scalable k-means++ algorithm
   * to select the initial cluster centers.
   *  - InitMethod::Random (random): Choose 'n_clusters' observations (rows) at
   * random from the input data for the initial centroids.
   *  - InitMethod::Array (ndarray): Use 'centroids' as initial cluster centers.
   */
  InitMethod init = KMeansPlusPlus;

  /**
   * Maximum number of iterations of the k-means algorithm for a single run.
   */
  int max_iter = 300;

  /**
   * Relative tolerance with regards to inertia to declare convergence.
   */
  double tol = 1e-4;

  /**
   * verbosity level.
   */
  int verbosity = RAFT_LEVEL_INFO;

  /**
   * Seed to the random number generator.
   */
  raft::random::RngState rng_state{0};

  /**
   * Number of instance k-means algorithm will be run with different seeds.
   */
  int n_init = 1;

  /**
   * Oversampling factor for use in the k-means|| algorithm
   */
  double oversampling_factor = 2.0;

  // batch_samples and batch_centroids are used to tile 1NN computation which is
  // useful to optimize/control the memory footprint
  // Default tile is [batch_samples x n_clusters] i.e. when batch_centroids is 0
  // then don't tile the centroids
  int batch_samples = 1 << 15;

  /**
   * if 0 then batch_centroids = n_clusters
   */
  int batch_centroids = 0;  //

  bool inertia_check = false;
};

/**
 * Simple object to specify hyper-parameters to the balanced k-means algorithm.
 *
 * The following metrics are currently supported in k-means balanced:
 *  - CosineExpanded
 *  - InnerProduct
 *  - L2Expanded
 *  - L2SqrtExpanded
 */
struct balanced_params : base_params {
  /**
   * Number of training iterations
   */
  uint32_t n_iters = 20;
};

/**
 * @}
 */

/**
 * @defgroup kmeans k-means clustering APIs
 * @{
 */

/**
 * @brief Find clusters with k-means algorithm.
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, params.n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               std::nullopt,
 *               centroids,
 *               raft::make_scalar_view(&inertia),
 *               raft::make_scalar_view(&n_iter));
 * @endcode
 *
 * @param[in]     handle        The raft handle.
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
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const float, int> X,
         std::optional<raft::device_vector_view<const float, int>> sample_weight,
         raft::device_matrix_view<float, int> centroids,
         raft::host_scalar_view<float, int> inertia,
         raft::host_scalar_view<int, int> n_iter);

/**
 * @brief Find clusters with k-means algorithm.
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int64_t n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<float, int64_t>(handle, params.n_clusters,
 * n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               std::nullopt,
 *               centroids,
 *               raft::make_scalar_view(&inertia),
 *               raft::make_scalar_view(&n_iter));
 * @endcode
 *
 * @param[in]     handle        The raft handle.
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
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const float, int64_t> X,
         std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float, int64_t> inertia,
         raft::host_scalar_view<int64_t, int64_t> n_iter);

/**
 * @brief Find clusters with k-means algorithm.
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<double, int>(handle, params.n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               std::nullopt,
 *               centroids,
 *               raft::make_scalar_view(&inertia),
 *               raft::make_scalar_view(&n_iter));
 * @endcode
 *
 * @param[in]     handle        The raft handle.
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
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const double, int> X,
         std::optional<raft::device_vector_view<const double, int>> sample_weight,
         raft::device_matrix_view<double, int> centroids,
         raft::host_scalar_view<double, int> inertia,
         raft::host_scalar_view<int, int> n_iter);

/**
 * @brief Find clusters with k-means algorithm.
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int64_t n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<double, int64_t>(handle, params.n_clusters,
 * n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               std::nullopt,
 *               centroids,
 *               raft::make_scalar_view(&inertia),
 *               raft::make_scalar_view(&n_iter));
 * @endcode
 *
 * @param[in]     handle        The raft handle.
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
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const double, int64_t> X,
         std::optional<raft::device_vector_view<const double, int64_t>> sample_weight,
         raft::device_matrix_view<double, int64_t> centroids,
         raft::host_scalar_view<double, int64_t> inertia,
         raft::host_scalar_view<int64_t, int64_t> n_iter);

/**
 * @brief Find clusters with k-means algorithm.
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, params.n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               std::nullopt,
 *               centroids,
 *               raft::make_scalar_view(&inertia),
 *               raft::make_scalar_view(&n_iter));
 * @endcode
 *
 * @param[in]     handle        The raft handle.
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
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const int8_t, int> X,
         std::optional<raft::device_vector_view<const int8_t, int>> sample_weight,
         raft::device_matrix_view<int8_t, int> centroids,
         raft::host_scalar_view<int8_t, int> inertia,
         raft::host_scalar_view<int, int> n_iter);

/**
 * @brief Find balanced clusters with k-means algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::balanced_params params;
 *   int n_features = 15;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, params.n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               centroids);
 * @endcode
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[out]  centroids       [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 */
void fit(const raft::resources& handle,
         cuvs::cluster::kmeans::balanced_params const& params,
         raft::device_matrix_view<const float, int> X,
         raft::device_matrix_view<float, int> centroids);

/**
 * @brief Find balanced clusters with k-means algorithm.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::balanced_params params;
 *   int n_features = 15;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, params.n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               centroids);
 * @endcode
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[inout]  centroids     [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 */
void fit(const raft::resources& handle,
         cuvs::cluster::kmeans::balanced_params const& params,
         raft::device_matrix_view<const int8_t, int> X,
         raft::device_matrix_view<int8_t, int> centroids);
/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, params.n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               std::nullopt,
 *               centroids.view(),
 *               raft::make_scalar_view(&inertia),
 *               raft::make_scalar_view(&n_iter));
 *   ...
 *   auto labels = raft::make_device_vector<int, int>(handle, X.extent(0));
 *
 *   kmeans::predict(handle,
 *                   params,
 *                   X,
 *                   std::nullopt,
 *                   centroids.view(),
 *                   false,
 *                   labels.view(),
 *                   raft::make_scalar_view(&ineratia));
 * @endcode
 *
 * @param[in]     handle           The raft handle.
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
void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const float, int> X,
             std::optional<raft::device_vector_view<const float, int>> sample_weight,
             raft::device_matrix_view<const float, int> centroids,
             raft::device_vector_view<int, int> labels,
             bool normalize_weight,
             raft::host_scalar_view<float> inertia);
void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const float, int> X,
             std::optional<raft::device_vector_view<const float, int>> sample_weight,
             raft::device_matrix_view<const float, int> centroids,
             raft::device_vector_view<int64_t, int> labels,
             bool normalize_weight,
             raft::host_scalar_view<float> inertia);

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<double, int>(handle, params.n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               std::nullopt,
 *               centroids.view(),
 *               raft::make_scalar_view(&inertia),
 *               raft::make_scalar_view(&n_iter));
 *   ...
 *   auto labels = raft::make_device_vector<int, int>(handle, X.extent(0));
 *
 *   kmeans::predict(handle,
 *                   params,
 *                   X,
 *                   std::nullopt,
 *                   centroids.view(),
 *                   false,
 *                   labels.view(),
 *                   raft::make_scalar_view(&ineratia));
 * @endcode
 *
 * @param[in]     handle           The raft handle.
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
void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const double, int> X,
             std::optional<raft::device_vector_view<const double, int>> sample_weight,
             raft::device_matrix_view<const double, int> centroids,
             raft::device_vector_view<int, int> labels,
             bool normalize_weight,
             raft::host_scalar_view<double> inertia);
void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const double, int> X,
             std::optional<raft::device_vector_view<const double, int>> sample_weight,
             raft::device_matrix_view<const double, int> centroids,
             raft::device_vector_view<int64_t, int> labels,
             bool normalize_weight,
             raft::host_scalar_view<double> inertia);

/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, params.n_clusters, n_features);
 *
 *   kmeans::fit(handle,
 *               params,
 *               X,
 *               std::nullopt,
 *               centroids.view(),
 *               raft::make_scalar_view(&inertia),
 *               raft::make_scalar_view(&n_iter));
 *   ...
 *   auto labels = raft::make_device_vector<int, int>(handle, X.extent(0));
 *
 *   kmeans::predict(handle,
 *                   params,
 *                   X,
 *                   std::nullopt,
 *                   centroids.view(),
 *                   false,
 *                   labels.view(),
 *                   raft::make_scalar_view(&ineratia));
 * @endcode
 *
 * @param[in]     handle           The raft handle.
 * @param[in]     params           Parameters for KMeans model.
 * @param[in]     X                New data to predict.
 *                                 [dim = n_samples x n_features]
 * @param[in]     centroids        Cluster centroids. The data must be in
 *                                 row-major format.
 *                                 [dim = n_clusters x n_features]
 * @param[out]    labels           Index of the cluster each sample in X
 *                                 belongs to.
 *                                 [len = n_samples]
 */
void predict(const raft::resources& handle,
             cuvs::cluster::kmeans::balanced_params const& params,
             raft::device_matrix_view<const int8_t, int> X,
             raft::device_matrix_view<const float, int> centroids,
             raft::device_vector_view<uint32_t, int> labels);

/**
 * @brief Compute k-means clustering and predicts cluster index for each sample
 * in the input.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, params.n_clusters, n_features);
 *   auto labels = raft::make_device_vector<int, int>(handle, X.extent(0));
 *
 *   kmeans::fit_predict(handle,
 *                       params,
 *                       X,
 *                       std::nullopt,
 *                       centroids.view(),
 *                       labels.view(),
 *                       raft::make_scalar_view(&inertia),
 *                       raft::make_scalar_view(&n_iter));
 * @endcode
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must be
 *                              in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional weights for each observation in X.
 *                              [len = n_samples]
 * @param[inout]  centroids     Optional
 *                              [in] When init is InitMethod::Array, use
 *                              centroids  as the initial cluster centers
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    labels        Index of the cluster each sample in X belongs
 *                              to.
 *                              [len = n_samples]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
void fit_predict(raft::resources const& handle,
                 const kmeans::params& params,
                 raft::device_matrix_view<const float, int> X,
                 std::optional<raft::device_vector_view<const float, int>> sample_weight,
                 std::optional<raft::device_matrix_view<float, int>> centroids,
                 raft::device_vector_view<int, int> labels,
                 raft::host_scalar_view<float> inertia,
                 raft::host_scalar_view<int> n_iter);
void fit_predict(raft::resources const& handle,
                 const kmeans::params& params,
                 raft::device_matrix_view<const float, int64_t> X,
                 std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
                 std::optional<raft::device_matrix_view<float, int64_t>> centroids,
                 raft::device_vector_view<int64_t, int64_t> labels,
                 raft::host_scalar_view<float> inertia,
                 raft::host_scalar_view<int64_t> n_iter);

/**
 * @brief Compute k-means clustering and predicts cluster index for each sample
 * in the input.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::params params;
 *   int n_features = 15, inertia, n_iter;
 *   auto centroids = raft::make_device_matrix<double, int>(handle, params.n_clusters, n_features);
 *   auto labels = raft::make_device_vector<int, int>(handle, X.extent(0));
 *
 *   kmeans::fit_predict(handle,
 *                       params,
 *                       X,
 *                       std::nullopt,
 *                       centroids.view(),
 *                       labels.view(),
 *                       raft::make_scalar_view(&inertia),
 *                       raft::make_scalar_view(&n_iter));
 * @endcode
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must be
 *                              in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[in]     sample_weight Optional weights for each observation in X.
 *                              [len = n_samples]
 * @param[inout]  centroids     Optional
 *                              [in] When init is InitMethod::Array, use
 *                              centroids  as the initial cluster centers
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    labels        Index of the cluster each sample in X belongs
 *                              to.
 *                              [len = n_samples]
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
void fit_predict(raft::resources const& handle,
                 const kmeans::params& params,
                 raft::device_matrix_view<const double, int> X,
                 std::optional<raft::device_vector_view<const double, int>> sample_weight,
                 std::optional<raft::device_matrix_view<double, int>> centroids,
                 raft::device_vector_view<int, int> labels,
                 raft::host_scalar_view<double> inertia,
                 raft::host_scalar_view<int> n_iter);
void fit_predict(raft::resources const& handle,
                 const kmeans::params& params,
                 raft::device_matrix_view<const double, int64_t> X,
                 std::optional<raft::device_vector_view<const double, int64_t>> sample_weight,
                 std::optional<raft::device_matrix_view<double, int64_t>> centroids,
                 raft::device_vector_view<int64_t, int64_t> labels,
                 raft::host_scalar_view<double> inertia,
                 raft::host_scalar_view<int64_t> n_iter);

/**
 * @brief Compute balanced k-means clustering and predicts cluster index for each sample
 * in the input.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::balanced_params params;
 *   int n_features = 15;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, params.n_clusters, n_features);
 *   auto labels = raft::make_device_vector<int, int>(handle, X.extent(0));
 *
 *   kmeans::fit_predict(handle,
 *                       params,
 *                       X,
 *                       centroids.view(),
 *                       labels.view());
 * @endcode
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must be
 *                              in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[inout]  centroids     Optional
 *                              [in] When init is InitMethod::Array, use
 *                              centroids  as the initial cluster centers
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    labels        Index of the cluster each sample in X belongs
 *                              to.
 *                              [len = n_samples]
 */
void fit_predict(const raft::resources& handle,
                 cuvs::cluster::kmeans::balanced_params const& params,
                 raft::device_matrix_view<const float, int> X,
                 raft::device_matrix_view<float, int> centroids,
                 raft::device_vector_view<uint32_t, int> labels);

/**
 * @brief Compute balanced k-means clustering and predicts cluster index for each sample
 * in the input.
 *
 * @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *   using namespace  cuvs::cluster;
 *   ...
 *   raft::resources handle;
 *   cuvs::cluster::kmeans::balanced_params params;
 *   int n_features = 15;
 *   auto centroids = raft::make_device_matrix<float, int>(handle, params.n_clusters, n_features);
 *   auto labels = raft::make_device_vector<int, int>(handle, X.extent(0));
 *
 *   kmeans::fit_predict(handle,
 *                       params,
 *                       X,
 *                       centroids.view(),
 *                       labels.view());
 * @endcode
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must be
 *                              in row-major format.
 *                              [dim = n_samples x n_features]
 * @param[inout]  centroids     Optional
 *                              [in] When init is InitMethod::Array, use
 *                              centroids  as the initial cluster centers
 *                              [out] The generated centroids from the
 *                              kmeans algorithm are stored at the address
 *                              pointed by 'centroids'.
 *                              [dim = n_clusters x n_features]
 * @param[out]    labels        Index of the cluster each sample in X belongs
 *                              to.
 *                              [len = n_samples]
 */
void fit_predict(const raft::resources& handle,
                 cuvs::cluster::kmeans::balanced_params const& params,
                 raft::device_matrix_view<const int8_t, int> X,
                 raft::device_matrix_view<float, int> centroids,
                 raft::device_vector_view<uint32_t, int> labels);

/**
 * @brief Transform X to a cluster-distance space.
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format
 *                              [dim = n_samples x n_features]
 * @param[in]     centroids     Cluster centroids. The data must be in row-major format.
 *                              [dim = n_clusters x n_features]
 * @param[out]    X_new         X transformed in the new space.
 *                              [dim = n_samples x n_features]
 */
void transform(raft::resources const& handle,
               const kmeans::params& params,
               raft::device_matrix_view<const float, int> X,
               raft::device_matrix_view<const float, int> centroids,
               raft::device_matrix_view<float, int> X_new);

/**
 * @brief Transform X to a cluster-distance space.
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 *                              be in row-major format
 *                              [dim = n_samples x n_features]
 * @param[in]     centroids     Cluster centroids. The data must be in row-major format.
 *                              [dim = n_clusters x n_features]
 * @param[out]    X_new         X transformed in the new space.
 *                              [dim = n_samples x n_features]
 */
void transform(raft::resources const& handle,
               const kmeans::params& params,
               raft::device_matrix_view<const double, int> X,
               raft::device_matrix_view<const double, int> centroids,
               raft::device_matrix_view<double, int> X_new);
/**
 * @}
 */

/**
 * @defgroup kmeans_helpers k-means API helpers
 * @{
 */

namespace helpers {

/**
 * Automatically find the optimal value of k using a binary search.
 * This method maximizes the Calinski-Harabasz Index while minimizing the per-cluster inertia.
 *
 *  @code{.cpp}
 *   #include <raft/core/resources.hpp>
 *   #include <cuvs/cluster/kmeans.hpp>
 *
 *   #include <raft/random/make_blobs.cuh>
 *
 *   using namespace  cuvs::cluster;
 *
 *   raft::handle_t handle;
 *   int n_samples = 100, n_features = 15, n_clusters = 10;
 *   auto X = raft::make_device_matrix<float, int>(handle, n_samples, n_features);
 *   auto labels = raft::make_device_vector<float, int>(handle, n_samples);
 *
 *   raft::random::make_blobs(handle, X, labels, n_clusters);
 *
 *   auto best_k = raft::make_host_scalar<int>(0);
 *   auto n_iter = raft::make_host_scalar<int>(0);
 *   auto inertia = raft::make_host_scalar<int>(0);
 *
 *   kmeans::find_k(handle, X, best_k.view(), inertia.view(), n_iter.view(), n_clusters+1);
 *
 * @endcode
 *
 * @param handle raft handle
 * @param X input observations (shape n_samples, n_dims)
 * @param best_k best k found from binary search
 * @param inertia inertia of best k found
 * @param n_iter number of iterations used to find best k
 * @param kmax maximum k to try in search
 * @param kmin minimum k to try in search (should be >= 1)
 * @param maxiter maximum number of iterations to run
 * @param tol tolerance for early stopping convergence
 */
void find_k(raft::resources const& handle,
            raft::device_matrix_view<const float, int> X,
            raft::host_scalar_view<int> best_k,
            raft::host_scalar_view<float> inertia,
            raft::host_scalar_view<int> n_iter,
            int kmax,
            int kmin    = 1,
            int maxiter = 100,
            float tol   = 1e-3);
}  // namespace helpers

/**
 * @}
 */

}  // namespace  cuvs::cluster::kmeans
