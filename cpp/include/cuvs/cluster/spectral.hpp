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

struct params {
  int n_clusters;
  int n_components;
  int n_init;
  int n_neighbors;
  float eigen_tolerance;
  raft::random::RngState rng_state{0};
};

// TODO: int64_t nnz support (see https://github.com/rapidsai/cuvs/issues/1484)
void fit_predict(raft::resources const& handle,
                 params config,
                 raft::device_coo_matrix_view<float, int, int, int> connectivity_graph,
                 raft::device_vector_view<int, int> labels);

void fit_predict(raft::resources const& handle,
                 params config,
                 raft::device_coo_matrix_view<double, int, int, int> connectivity_graph,
                 raft::device_vector_view<int, int> labels);

}  // namespace cuvs::cluster::spectral
