/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vpq_dataset.cuh"

namespace cuvs::neighbors::detail {

#define PROCESS_AND_FILL_CODES_SUBSPACES_IMPL(MathT, IdxT, DatasetT)             \
  auto process_and_fill_codes_subspaces(                                         \
    const raft::resources& res,                                                  \
    const vpq_params& params,                                                    \
    DatasetT dataset,                                                            \
    raft::device_matrix_view<const MathT, uint32_t, raft::row_major> vq_centers, \
    raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers) \
    -> raft::device_matrix<uint8_t, IdxT, raft::row_major>                       \
  {                                                                              \
    return process_and_fill_codes_subspaces<MathT, IdxT, DatasetT>(              \
      res, params, dataset, vq_centers, pq_centers);                             \
  }

#define COMMA ,

PROCESS_AND_FILL_CODES_SUBSPACES_IMPL(
  float, int64_t, raft::device_matrix_view<const float COMMA int64_t COMMA raft::row_major>)
PROCESS_AND_FILL_CODES_SUBSPACES_IMPL(
  double, int64_t, raft::device_matrix_view<const double COMMA int64_t COMMA raft::row_major>)

#undef PROCESS_AND_FILL_CODES_SUBSPACES_IMPL

}  // namespace cuvs::neighbors::detail
