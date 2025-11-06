/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/spectral_embedding.cuh"

#include <cuvs/preprocessing/spectral_embedding.hpp>

namespace cuvs::preprocessing::spectral_embedding {

#define CUVS_INST_SPECTRAL_EMBEDDING(DataT)                                             \
  void transform(raft::resources const& handle,                                         \
                 params config,                                                         \
                 raft::device_coo_matrix_view<DataT, int, int, int> connectivity_graph, \
                 raft::device_matrix_view<DataT, int, raft::col_major> embedding)       \
  {                                                                                     \
    detail::transform<DataT>(handle, config, connectivity_graph, embedding);            \
  }

CUVS_INST_SPECTRAL_EMBEDDING(float);
CUVS_INST_SPECTRAL_EMBEDDING(double);

#undef CUVS_INST_SPECTRAL_EMBEDDING

// Non-template functions
void transform(raft::resources const& handle,
               params config,
               raft::device_matrix_view<float, int, raft::row_major> dataset,
               raft::device_matrix_view<float, int, raft::col_major> embedding)
{
  detail::transform(handle, config, dataset, embedding);
}

}  // namespace cuvs::preprocessing::spectral_embedding

namespace cuvs::preprocessing::spectral_embedding::helpers {

void create_connectivity_graph(raft::resources const& handle,
                               params spectral_embedding_config,
                               raft::device_matrix_view<float, int, raft::row_major> dataset,
                               raft::device_coo_matrix<float, int, int, int>& connectivity_graph)
{
  detail::create_connectivity_graph(handle, spectral_embedding_config, dataset, connectivity_graph);
}

}  // namespace cuvs::preprocessing::spectral_embedding::helpers
