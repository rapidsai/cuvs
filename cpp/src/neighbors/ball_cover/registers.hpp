/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "registers_types.cuh"
#include <cuvs/neighbors/ball_cover.hpp>

#include <raft/core/resources.hpp>

namespace cuvs::neighbors::ball_cover::detail {

template <typename value_idx,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t>
void rbc_low_dim_pass_one(
  raft::resources const& handle,
  const cuvs::neighbors::ball_cover::index<value_idx, value_t, value_int, matrix_idx>& index,
  const value_t* query,
  const value_int n_query_rows,
  value_int k,
  const value_idx* R_knn_inds,
  const value_t* R_knn_dists,
  value_idx* inds,
  value_t* dists,
  float weight,
  value_int* dists_counter,
  int dims);

template <typename value_idx,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t>
void rbc_low_dim_pass_two(
  raft::resources const& handle,
  const cuvs::neighbors::ball_cover::index<value_idx, value_t, value_int, matrix_idx>& index,
  const value_t* query,
  const value_int n_query_rows,
  value_int k,
  const value_idx* R_knn_inds,
  const value_t* R_knn_dists,
  value_idx* inds,
  value_t* dists,
  float weight,
  value_int* post_dists_counter,
  int dims);

template <typename value_idx,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t,
          typename dist_func>
void rbc_eps_pass(
  raft::resources const& handle,
  const cuvs::neighbors::ball_cover::index<value_idx, value_t, value_int, matrix_idx>& index,
  const value_t* query,
  const value_int n_query_rows,
  value_t eps,
  const value_t* R,
  dist_func& dfunc,
  bool* adj,
  value_idx* vd);

template <typename value_idx,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t,
          typename dist_func>
void rbc_eps_pass(
  raft::resources const& handle,
  const cuvs::neighbors::ball_cover::index<value_idx, value_t, value_int, matrix_idx>& index,
  const value_t* query,
  const value_int n_query_rows,
  value_t eps,
  value_int* max_k,
  const value_t* R,
  dist_func& dfunc,
  value_idx* adj_ia,
  value_idx* adj_ja,
  value_idx* vd);

}  // namespace cuvs::neighbors::ball_cover::detail

extern template void
cuvs::neighbors::ball_cover::detail::rbc_low_dim_pass_one<int64_t, float, int64_t, int64_t>(
  raft::resources const& handle,
  const cuvs::neighbors::ball_cover::index<int64_t, float, int64_t, int64_t>& index,
  const float* query,
  const int64_t n_query_rows,
  int64_t k,
  const int64_t* R_knn_inds,
  const float* R_knn_dists,
  int64_t* inds,
  float* dists,
  float weight,
  int64_t* dists_counter,
  int dims);

extern template void
cuvs::neighbors::ball_cover::detail::rbc_low_dim_pass_two<int64_t, float, int64_t, int64_t>(
  raft::resources const& handle,
  const cuvs::neighbors::ball_cover::index<int64_t, float, int64_t, int64_t>& index,
  const float* query,
  const int64_t n_query_rows,
  int64_t k,
  const int64_t* R_knn_inds,
  const float* R_knn_dists,
  int64_t* inds,
  float* dists,
  float weight,
  int64_t* post_dists_counter,
  int dims);

extern template void cuvs::neighbors::ball_cover::detail::rbc_eps_pass<
  int64_t,
  float,
  int64_t,
  int64_t,
  cuvs::neighbors::ball_cover::detail::EuclideanFunc<float, int64_t>>(
  raft::resources const& handle,
  const cuvs::neighbors::ball_cover::index<int64_t, float, int64_t, int64_t>& index,
  const float* query,
  const int64_t n_query_rows,
  float eps,
  const float* R,
  cuvs::neighbors::ball_cover::detail::EuclideanFunc<float, int64_t>& dfunc,
  bool* adj,
  int64_t* vd);

extern template void cuvs::neighbors::ball_cover::detail::rbc_eps_pass<
  int64_t,
  float,
  int64_t,
  int64_t,
  cuvs::neighbors::ball_cover::detail::EuclideanFunc<float, int64_t>>(
  raft::resources const& handle,
  const cuvs::neighbors::ball_cover::index<int64_t, float, int64_t, int64_t>& index,
  const float* query,
  const int64_t n_query_rows,
  float eps,
  const float* R,
  cuvs::neighbors::ball_cover::detail::EuclideanFunc<float, int64_t>& dfunc,
  bool* adj,
  int64_t* vd);
