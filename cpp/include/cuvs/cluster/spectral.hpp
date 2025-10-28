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
  raft::random::RngState rng_state{0};
};

template <typename DataT>
void fit_predict(raft::resources const& handle,
                 params config,
                 raft::device_coo_matrix_view<DataT, int, int, int> connectivity_graph,
                 raft::device_vector_view<int, int> labels);

}  // namespace cuvs::cluster::spectral
