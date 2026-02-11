/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "registers_types.cuh"
#include <cuvs/neighbors/ball_cover.hpp>

#include <raft/core/resources.hpp>

namespace cuvs::neighbors::ball_cover::detail {

template <typename ValueIdx, typename ValueT>
void rbc_low_dim_pass_one(raft::resources const& handle,
                          const cuvs::neighbors::ball_cover::index<ValueIdx, ValueT>& index,
                          const ValueT* query,
                          const int64_t n_query_rows,
                          const int64_t k,
                          const ValueIdx* R_knn_inds,
                          const ValueT* R_knn_dists,
                          ValueIdx* inds,
                          ValueT* dists,
                          float weight,
                          int dims);

template <typename ValueIdx, typename ValueT>
void rbc_low_dim_pass_two(raft::resources const& handle,
                          const cuvs::neighbors::ball_cover::index<ValueIdx, ValueT>& index,
                          const ValueT* query,
                          const int64_t n_query_rows,
                          const int64_t k,
                          const ValueIdx* R_knn_inds,
                          const ValueT* R_knn_dists,
                          ValueIdx* inds,
                          ValueT* dists,
                          float weight,
                          int dims);

template <typename ValueIdx, typename ValueT, typename dist_func>
void rbc_eps_pass(raft::resources const& handle,
                  const cuvs::neighbors::ball_cover::index<ValueIdx, ValueT>& index,
                  const ValueT* query,
                  const int64_t n_query_rows,
                  ValueT eps,
                  const ValueT* R,
                  dist_func& dfunc,
                  bool* adj,
                  ValueIdx* vd);

template <typename ValueIdx, typename ValueT, typename dist_func>
void rbc_eps_pass(raft::resources const& handle,
                  const cuvs::neighbors::ball_cover::index<ValueIdx, ValueT>& index,
                  const ValueT* query,
                  const int64_t n_query_rows,
                  ValueT eps,
                  int64_t* max_k,
                  const ValueT* R,
                  dist_func& dfunc,
                  ValueIdx* adj_ia,
                  ValueIdx* adj_ja,
                  ValueIdx* vd);

}  // namespace cuvs::neighbors::ball_cover::detail
