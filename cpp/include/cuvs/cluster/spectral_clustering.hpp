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

#include <cstdint>
#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::cluster::spectral_clustering {

struct params {
  int n_clusters;
  int n_components;
  int n_init;
  int n_neighbors;
  uint64_t seed;
};

void fit_predict(raft::resources const& handle,
                 params config,
                 raft::device_coo_matrix_view<float, int, int, int> connectivity_graph,
                 raft::device_vector_view<int, int> labels);

}  // namespace cuvs::cluster::spectral_clustering
