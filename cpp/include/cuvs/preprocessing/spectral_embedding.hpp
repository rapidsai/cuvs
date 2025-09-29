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

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::preprocessing::spectral_embedding {

/**
 * @brief Parameters for spectral embedding algorithm
 *
 * Spectral embedding is a dimensionality reduction technique that uses the
 * eigenvectors of the graph Laplacian to embed data points into a lower-dimensional
 * space. This technique is particularly useful for non-linear dimensionality
 * reduction and clustering tasks.
 */
struct params {
  /** @brief The number of components to reduce the data to. */
  int n_components;

  /** @brief The number of neighbors to use for the nearest neighbors graph. */
  int n_neighbors;

  /**
   * @brief Whether to normalize the Laplacian matrix.
   *
   * If true, uses the normalized graph Laplacian (D^(-1/2) L D^(-1/2)).
   * If false, uses the unnormalized graph Laplacian (L = D - W).
   * Normalized Laplacian often leads to better results for clustering tasks.
   */
  bool norm_laplacian;

  /**
   * @brief Whether to drop the first eigenvector.
   *
   * The first eigenvector of the normalized Laplacian is constant and
   * uninformative. Setting this to true drops it from the embedding.
   * This is typically set to true when norm_laplacian is true.
   */
  bool drop_first;

  /**
   * @brief Random seed for reproducibility.
   *
   * Controls the random number generation for k-NN graph construction
   * and eigenvalue solver initialization. Use the same seed value to
   * ensure reproducible results across runs.
   */
  uint64_t seed;
};

/**
 * @defgroup spectral_embedding Spectral Embedding
 * @{
 */

/**
 * @brief Perform spectral embedding on input dataset
 *
 * This function computes the spectral embedding of the input dataset by:
 * 1. Constructing a k-nearest neighbors graph from the input data
 * 2. Computing the graph Laplacian (normalized or unnormalized)
 * 3. Finding the eigenvectors corresponding to the smallest eigenvalues
 * 4. Using these eigenvectors as the embedding coordinates
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/preprocessing/spectral_embedding.hpp>
 *
 * raft::resources handle;
 *
 * // Set up parameters
 * cuvs::preprocessing::spectral_embedding::params params;
 * params.n_components = 2;
 * params.n_neighbors = 15;
 * params.norm_laplacian = true;
 * params.drop_first = true;
 * params.seed = 42;
 *
 * // Create input dataset (n_samples x n_features)
 * auto dataset = raft::make_device_matrix<float, int>(handle, n_samples, n_features);
 * // ... fill dataset ...
 *
 * // Create output embedding matrix (n_samples x n_components)
 * auto embedding = raft::make_device_matrix<float, int, raft::col_major>(
 *     handle, n_samples, params.n_components);
 *
 * // Perform spectral embedding
 * cuvs::preprocessing::spectral_embedding::transform(
 *     handle, params, dataset.view(), embedding.view());
 * @endcode
 *
 * @param[in] handle RAFT resource handle for managing CUDA resources
 * @param[in] config Parameters controlling the spectral embedding algorithm
 * @param[in] dataset Input dataset in row-major format [n_samples x n_features]
 * @param[out] embedding Output embedding in column-major format [n_samples x n_components]
 *
 */
void transform(raft::resources const& handle,
               params config,
               raft::device_matrix_view<float, int, raft::row_major> dataset,
               raft::device_matrix_view<float, int, raft::col_major> embedding);

/**
 * @brief Perform spectral embedding using a precomputed connectivity graph
 *
 * This function computes the spectral embedding from a precomputed sparse
 * connectivity graph (e.g., from a k-NN search or custom similarity matrix).
 * This is useful when you want to use a custom graph construction method
 * or when you have a precomputed similarity/affinity matrix.
 *
 * The function:
 * 1. Converts the COO matrix to the graph Laplacian
 * 2. Computes eigenvectors of the Laplacian
 * 3. Returns the eigenvectors as the embedding
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/preprocessing/spectral_embedding.hpp>
 *
 * raft::resources handle;
 *
 * // Set up parameters
 * cuvs::preprocessing::spectral_embedding::params params;
 * params.n_components = 2;
 * params.norm_laplacian = true;
 * params.drop_first = true;
 * params.seed = 42;
 *
 * // Assume we have a precomputed connectivity graph as COO matrix
 * // connectivity_graph represents weighted edges between samples
 * raft::device_coo_matrix<float, int, int, int> connectivity_graph(...);
 *
 * // Create output embedding matrix (n_samples x n_components)
 * auto embedding = raft::make_device_matrix<float, int, raft::col_major>(
 *     handle, n_samples, params.n_components);
 *
 * // Perform spectral embedding
 * cuvs::preprocessing::spectral_embedding::transform(
 *     handle, params, connectivity_graph.view(), embedding.view());
 * @endcode
 *
 * @param[in] handle RAFT resource handle for managing CUDA resources
 * @param[in] config Parameters controlling the spectral embedding algorithm
 *                   (n_neighbors parameter is ignored when using precomputed graph)
 * @param[in] connectivity_graph Precomputed sparse connectivity/affinity graph in COO format
 *                               representing weighted connections between samples
 * @param[out] embedding Output embedding in column-major format [n_samples x n_components]
 *
 */
void transform(raft::resources const& handle,
               params config,
               raft::device_coo_matrix_view<float, int, int, int> connectivity_graph,
               raft::device_matrix_view<float, int, raft::col_major> embedding);

/**
 * @}
 */

}  // namespace cuvs::preprocessing::spectral_embedding
