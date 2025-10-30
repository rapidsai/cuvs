/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/spectral.cuh"

#include <cuvs/cluster/spectral.hpp>

namespace cuvs::cluster::spectral {

#define CUVS_INST_SPECTRAL(DataT)                                                         \
  void fit_predict(raft::resources const& handle,                                         \
                   params config,                                                         \
                   raft::device_coo_matrix_view<DataT, int, int, int> connectivity_graph, \
                   raft::device_vector_view<int, int> labels)                             \
  {                                                                                       \
    detail::fit_predict<DataT>(handle, config, connectivity_graph, labels);               \
  }

CUVS_INST_SPECTRAL(float);
CUVS_INST_SPECTRAL(double);

#undef CUVS_INST_SPECTRAL

}  // namespace cuvs::cluster::spectral
