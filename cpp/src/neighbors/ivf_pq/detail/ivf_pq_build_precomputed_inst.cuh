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
  template <typename pq_centers_accessor,                                                          \
            typename centers_accessor,                                                             \
            typename centers_rot_accessor,                                                         \
            typename rotation_matrix_accessor,                                                     \
            typename = std::enable_if_t<                                                           \
              raft::is_device_mdspan_v<raft::mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major, pq_centers_accessor>>>> \
  auto build(                                                                                      \
    raft::resources const& handle,                                                                 \
    const cuvs::neighbors::ivf_pq::index_params& index_params,                                     \
    const uint32_t dim,                                                                            \
    raft::mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major, pq_centers_accessor> pq_centers, \
    raft::mdspan<const float, raft::matrix_extent<uint32_t>, raft::row_major, centers_accessor> centers, \
    raft::mdspan<const float, raft::matrix_extent<uint32_t>, raft::row_major, centers_rot_accessor> centers_rot, \
    raft::mdspan<const float, raft::matrix_extent<uint32_t>, raft::row_major, rotation_matrix_accessor> rotation_matrix) \
    -> cuvs::neighbors::ivf_pq::index<IdxT>                                                        \
  {                                                                                                \
    return detail::build<IdxT>(                                                                    \
      handle, index_params, dim, pq_centers, centers, centers_rot, rotation_matrix);               \
  }                                                                                                \

}  // namespace cuvs::neighbors::ivf_pq
