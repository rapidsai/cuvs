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

// Template function declarations
template <typename IndexTypeT>
void transform(raft::resources const& handle,
               params config,
               raft::device_matrix_view<float, IndexTypeT, raft::row_major> dataset,
               raft::device_matrix_view<float, IndexTypeT, raft::col_major> embedding);

template <typename IndexTypeT>
void transform(
  raft::resources const& handle,
  params config,
  raft::device_coo_matrix_view<float, IndexTypeT, IndexTypeT, IndexTypeT> connectivity_graph,
  raft::device_matrix_view<float, IndexTypeT, raft::col_major> embedding);

// Explicit instantiations for common index types
extern template void transform<int32_t>(
  raft::resources const& handle,
  params config,
  raft::device_matrix_view<float, int32_t, raft::row_major> dataset,
  raft::device_matrix_view<float, int32_t, raft::col_major> embedding);

extern template void transform<int64_t>(
  raft::resources const& handle,
  params config,
  raft::device_matrix_view<float, int64_t, raft::row_major> dataset,
  raft::device_matrix_view<float, int64_t, raft::col_major> embedding);

extern template void transform<int32_t>(
  raft::resources const& handle,
  params config,
  raft::device_coo_matrix_view<float, int32_t, int32_t, int32_t> connectivity_graph,
  raft::device_matrix_view<float, int32_t, raft::col_major> embedding);

extern template void transform<int64_t>(
  raft::resources const& handle,
  params config,
  raft::device_coo_matrix_view<float, int64_t, int64_t, int64_t> connectivity_graph,
  raft::device_matrix_view<float, int64_t, raft::col_major> embedding);

}  // namespace cuvs::preprocessing::spectral_embedding
