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

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/cluster/spectral.hpp>
#include <cuvs/preprocessing/spectral_embedding.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/random/rng_state.hpp>

namespace cuvs::cluster::spectral {

void fit_predict(raft::resources const& handle,
                 params config,
                 raft::device_coo_matrix_view<float, int, int, int> connectivity_graph,
                 raft::device_vector_view<int, int> labels)
{
  int n_samples = connectivity_graph.structure_view().get_n_rows();
  float inertia;
  int n_iter;
  auto embedding_col_major =
    raft::make_device_matrix<float, int, raft::col_major>(handle, n_samples, config.n_components);
  auto embedding_row_major =
    raft::make_device_matrix<float, int, raft::row_major>(handle, n_samples, config.n_components);
  cuvs::preprocessing::spectral_embedding::params spectral_embedding_config;
  spectral_embedding_config.n_components   = config.n_components;
  spectral_embedding_config.n_neighbors    = config.n_neighbors;
  spectral_embedding_config.norm_laplacian = true;
  spectral_embedding_config.drop_first     = false;
  spectral_embedding_config.seed           = config.seed;

  cuvs::cluster::kmeans::params kmeans_config;
  kmeans_config.n_clusters          = config.n_clusters;
  kmeans_config.rng_state           = raft::random::RngState(config.seed);
  kmeans_config.n_init              = config.n_init;
  kmeans_config.oversampling_factor = 0.0;

  cuvs::preprocessing::spectral_embedding::transform(
    handle, spectral_embedding_config, connectivity_graph, embedding_col_major.view());

  raft::linalg::transpose(handle,
                          embedding_col_major.data_handle(),
                          embedding_row_major.data_handle(),
                          n_samples,
                          config.n_components,
                          raft::resource::get_cuda_stream(handle));

  cuvs::cluster::kmeans::fit_predict(handle,
                                     kmeans_config,
                                     embedding_row_major.view(),
                                     std::nullopt,
                                     std::nullopt,
                                     labels,
                                     raft::make_host_scalar_view(&inertia),
                                     raft::make_host_scalar_view(&n_iter));
}
}  // namespace cuvs::cluster::spectral
