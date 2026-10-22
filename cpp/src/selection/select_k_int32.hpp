/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cuvs/selection/select_k.hpp>

namespace cuvs::selection {
__attribute__((visibility("hidden"))) void select_k(
  raft::resources const& handle,
  raft::device_matrix_view<const int32_t, int64_t, raft::row_major> in_val,
  std::optional<raft::device_matrix_view<const uint32_t, int64_t, raft::row_major>> in_idx,
  raft::device_matrix_view<int32_t, int64_t, raft::row_major> out_val,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> out_idx,
  bool select_min,
  bool sorted = false,
  SelectAlgo algo = SelectAlgo::kAuto,
  std::optional<raft::device_vector_view<const uint32_t, int64_t>> len_i = std::nullopt);
}  // namespace cuvs::selection
