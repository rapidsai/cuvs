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
 */
struct params {
  /** @brief The number of components to reduce the data to. */
  int n_components;

  /** @brief The number of neighbors to use for the nearest neighbors graph. */
  int n_neighbors;

  /** @brief Whether to normalize the Laplacian matrix. */
  bool norm_laplacian;

  /** @brief Whether to drop the first eigenvector. */
  bool drop_first;

  /** @brief Random seed for reproducibility */
  uint64_t seed;
};

void create_connectivity_graph(raft::resources const& handle,
                               params spectral_embedding_config,
                               raft::device_matrix_view<float, int, raft::row_major> dataset,
                               raft::device_matrix_view<float, int, raft::col_major> embedding,
                               raft::device_coo_matrix<float, int, int, int>& connectivity_graph);

raft::device_csr_matrix_view<float, int, int, int> coo_to_csr_matrix(
  raft::resources const& handle,
  const int n_samples,
  raft::device_vector_view<int> sym_coo_row_ind,
  raft::device_coo_matrix<float, int, int, int>& sym_coo_matrix);

raft::device_csr_matrix<float, int, int, int> create_laplacian(
  raft::resources const& handle,
  params spectral_embedding_config,
  raft::device_csr_matrix_view<float, int, int, int> csr_matrix_view,
  raft::device_vector_view<float, int> diagonal);

void compute_eigenpairs(raft::resources const& handle,
                        params spectral_embedding_config,
                        const int n_samples,
                        raft::device_csr_matrix<float, int, int, int> laplacian,
                        raft::device_vector_view<float, int> diagonal,
                        raft::device_matrix_view<float, int, raft::col_major> embedding);

void transform(raft::resources const& handle,
               params config,
               raft::device_matrix_view<float, int, raft::row_major> dataset,
               raft::device_matrix_view<float, int, raft::col_major> embedding);

}  // namespace cuvs::preprocessing::spectral_embedding
