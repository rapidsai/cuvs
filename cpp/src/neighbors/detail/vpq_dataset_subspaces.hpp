/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::neighbors::detail {

auto process_and_fill_codes_subspaces(
  const raft::resources& res,
  const vpq_params& params,
  raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
  raft::device_matrix_view<const float, uint32_t, raft::row_major> vq_centers,
  raft::device_matrix_view<const float, uint32_t, raft::row_major> pq_centers)
  -> raft::device_matrix<uint8_t, int64_t, raft::row_major>;

auto process_and_fill_codes_subspaces(
  const raft::resources& res,
  const vpq_params& params,
  raft::device_matrix_view<const double, int64_t, raft::row_major> dataset,
  raft::device_matrix_view<const double, uint32_t, raft::row_major> vq_centers,
  raft::device_matrix_view<const double, uint32_t, raft::row_major> pq_centers)
  -> raft::device_matrix<uint8_t, int64_t, raft::row_major>;

}  // namespace cuvs::neighbors::detail
