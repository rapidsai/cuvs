/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/ivf_pq.hpp>

#include "../ivf_pq_build.cuh"

namespace cuvs::neighbors::ivf_pq {
#define CUVS_INST_IVF_PQ_BUILD_PRECOMPUTED(IdxT)                                                   \
  auto build(                                                                                      \
    raft::resources const& handle,                                                                 \
    const cuvs::neighbors::ivf_pq::index_params& index_params,                                     \
    const uint32_t dim,                                                                            \
    raft::host_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,         \
    raft::host_matrix_view<const float, uint32_t, raft::row_major> centers,                        \
    std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>> centers_rot,     \
    std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>> rotation_matrix) \
    -> cuvs::neighbors::ivf_pq::index<IdxT>                                                        \
  {                                                                                                \
    return detail::build<IdxT>(                                                                    \
      handle, index_params, dim, pq_centers, centers, centers_rot, rotation_matrix);               \
  }                                                                                                \
  void build(                                                                                      \
    raft::resources const& handle,                                                                 \
    const cuvs::neighbors::ivf_pq::index_params& index_params,                                     \
    const uint32_t dim,                                                                            \
    raft::host_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,         \
    raft::host_matrix_view<const float, uint32_t, raft::row_major> centers,                        \
    std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>> centers_rot,     \
    std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>> rotation_matrix, \
    cuvs::neighbors::ivf_pq::index<IdxT>* idx)                                                     \
  {                                                                                                \
    detail::build<IdxT>(                                                                           \
      handle, index_params, dim, pq_centers, centers, centers_rot, rotation_matrix, idx);          \
  }                                                                                                \
  auto build(                                                                                      \
    raft::resources const& handle,                                                                 \
    const cuvs::neighbors::ivf_pq::index_params& index_params,                                     \
    const uint32_t dim,                                                                            \
    raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,       \
    raft::device_matrix_view<const float, uint32_t, raft::row_major> centers,                      \
    raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot,                  \
    raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix)              \
    -> cuvs::neighbors::ivf_pq::index<IdxT>                                                        \
  {                                                                                                \
    return detail::build<IdxT>(                                                                    \
      handle, index_params, dim, pq_centers, centers, centers_rot, rotation_matrix);               \
  }                                                                                                \

}  // namespace cuvs::neighbors::ivf_pq
