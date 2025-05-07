#include "registers.cuh"
#include <cuvs/neighbors/ball_cover.hpp>

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
