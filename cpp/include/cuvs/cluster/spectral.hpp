/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng_state.hpp>

namespace cuvs::cluster::spectral {

/**
 * @defgroup spectral_params Spectral Clustering Parameters
 * @{
 */

/**
 * @brief Parameters for spectral clustering
 */
struct params {
  /** Number of clusters to find */
  int n_clusters;

  /** Number of eigenvectors to use for the spectral embedding (typically equal to n_clusters) */
  int n_components;

  /** Number of k-means runs with different centroid seeds */
  int n_init;

  /** Number of nearest neighbors for constructing the connectivity graph */
  int n_neighbors;

  /** Random number generator state for reproducibility */
  raft::random::RngState rng_state{0};
};

/** @} */  // end of spectral_params group

/**
 * @defgroup spectral Spectral Clustering
 * @{
 */

// TODO: int64_t nnz support (see https://github.com/rapidsai/cuvs/issues/1484)

/**
 * @brief Perform spectral clustering on a connectivity graph
 *
 * @param[in] handle RAFT resource handle
 * @param[in] config Spectral clustering parameters
 * @param[in] connectivity_graph Sparse COO matrix representing connectivity between data points
 * @param[out] labels Device vector of size n_samples to store cluster assignments (0 to
 * n_clusters-1)
 *
 * @code{.cpp}
 * #include <cuvs/cluster/spectral.hpp>
 * #include <cuvs/preprocessing/spectral_embedding.hpp>
 *
 * raft::resources handle;
 *
 * // Create connectivity graph from data
 * auto graph = raft::make_device_coo_matrix<float>(handle, n_samples, n_samples);
 * cuvs::preprocessing::spectral_embedding::params embed_params;
 * embed_params.n_neighbors = 15;
 * cuvs::preprocessing::spectral_embedding::helpers::create_connectivity_graph(
 *     handle, embed_params, X.view(), graph);
 *
 * // Configure and run spectral clustering
 * cuvs::cluster::spectral::params params;
 * params.n_clusters = 5;
 * params.n_components = 5;
 * params.n_neighbors = 15;
 * params.n_init = 10;
 *
 * auto labels = raft::make_device_vector<int>(handle, n_samples);
 * cuvs::cluster::spectral::fit_predict(handle, params, graph.view(), labels.view());
 * @endcode
 */
void fit_predict(raft::resources const& handle,
                 params config,
                 raft::device_coo_matrix_view<float, int, int, int> connectivity_graph,
                 raft::device_vector_view<int, int> labels);

/**
 * @brief Perform spectral clustering on a connectivity graph
 *
 * @param[in] handle RAFT resource handle
 * @param[in] config Spectral clustering parameters
 * @param[in] connectivity_graph Sparse COO matrix representing connectivity between data points
 * @param[out] labels Device vector of size n_samples to store cluster assignments (0 to
 * n_clusters-1)
 *
 * @code{.cpp}
 * #include <cuvs/cluster/spectral.hpp>
 * #include <cuvs/preprocessing/spectral_embedding.hpp>
 *
 * raft::resources handle;
 *
 * // Create connectivity graph from data
 * auto graph = raft::make_device_coo_matrix<double>(handle, n_samples, n_samples);
 * cuvs::preprocessing::spectral_embedding::params embed_params;
 * embed_params.n_neighbors = 15;
 * cuvs::preprocessing::spectral_embedding::helpers::create_connectivity_graph(
 *     handle, embed_params, X_double.view(), graph);
 *
 * // Configure and run spectral clustering
 * cuvs::cluster::spectral::params params;
 * params.n_clusters = 5;
 * params.n_components = 5;
 * params.n_neighbors = 15;
 * params.n_init = 10;
 *
 * auto labels = raft::make_device_vector<int>(handle, n_samples);
 * cuvs::cluster::spectral::fit_predict(handle, params, graph.view(), labels.view());
 * @endcode
 */
void fit_predict(raft::resources const& handle,
                 params config,
                 raft::device_coo_matrix_view<double, int, int, int> connectivity_graph,
                 raft::device_vector_view<int, int> labels);

/**
 * @}
 */

}  // namespace cuvs::cluster::spectral
