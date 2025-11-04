/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../refine_host.hpp"
#include <cuvs/neighbors/refine.hpp>

#include <cuda_fp16.h>

#define instantiate_cuvs_neighbors_refine(idx_t, data_t, distance_t, matrix_idx)          \
  void cuvs::neighbors::refine(                                                           \
    raft::resources const& handle,                                                        \
    raft::host_matrix_view<const data_t, matrix_idx, raft::row_major> dataset,            \
    raft::host_matrix_view<const data_t, matrix_idx, raft::row_major> queries,            \
    raft::host_matrix_view<const idx_t, matrix_idx, raft::row_major> neighbor_candidates, \
    raft::host_matrix_view<idx_t, matrix_idx, raft::row_major> indices,                   \
    raft::host_matrix_view<distance_t, matrix_idx, raft::row_major> distances,            \
    cuvs::distance::DistanceType metric)                                                  \
  {                                                                                       \
    refine_impl<idx_t, data_t, distance_t, matrix_idx>(                                   \
      handle, dataset, queries, neighbor_candidates, indices, distances, metric);         \
  }

instantiate_cuvs_neighbors_refine(int64_t, half, float, int64_t);

#undef instantiate_cuvs_neighbors_refine
