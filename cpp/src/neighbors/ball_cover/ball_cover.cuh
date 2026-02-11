/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"
#include "registers.hpp"
#include <cuvs/neighbors/ball_cover.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/copy.cuh>
#include <raft/random/rng.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <climits>

#include <cstdint>

namespace cuvs::neighbors::ball_cover::detail {

/**
 * Given a set of points in row-major order which are to be
 * used as a set of index points, uniformly samples a subset
 * of points to be used as landmarks.
 * @tparam ValueIdx
 * @tparam ValueT
 * @param handle
 * @param index
 */
template <typename ValueIdx, typename ValueT>
void sample_landmarks(raft::resources const& handle,
                      cuvs::neighbors::ball_cover::index<ValueIdx, ValueT>& index)
{
  rmm::device_uvector<ValueIdx> r_1nn_cols2(index.n_landmarks,
                                            raft::resource::get_cuda_stream(handle));
  rmm::device_uvector<ValueT> r_1nn_ones(index.m, raft::resource::get_cuda_stream(handle));
  rmm::device_uvector<ValueIdx> r_indices(index.n_landmarks,
                                          raft::resource::get_cuda_stream(handle));

  raft::linalg::map_offset(handle, index.get_R_1nn_cols(), raft::identity_op{});

  thrust::fill(raft::resource::get_thrust_policy(handle),
               r_1nn_ones.data(),
               r_1nn_ones.data() + r_1nn_ones.size(),
               1.0);

  thrust::fill(raft::resource::get_thrust_policy(handle),
               r_indices.data(),
               r_indices.data() + r_indices.size(),
               0.0);

  /**
   * 1. Randomly sample sqrt(n) points from X
   */
  raft::random::RngState rng_state(12345);
  raft::random::sampleWithoutReplacement(handle,
                                         rng_state,
                                         r_indices.data(),
                                         r_1nn_cols2.data(),
                                         index.get_R_1nn_cols().data_handle(),
                                         r_1nn_ones.data(),
                                         static_cast<ValueIdx>(index.n_landmarks),
                                         static_cast<ValueIdx>(index.m));

  auto x = index.get_X();
  auto r = index.get_R();

  raft::matrix::copy_rows<ValueT, ValueIdx>(
    handle,
    raft::make_device_matrix_view<const ValueT, ValueIdx>(
      x.data_handle(), x.extent(0), x.extent(1)),
    raft::make_device_matrix_view<ValueT, ValueIdx>(r.data_handle(), r.extent(0), r.extent(1)),
    raft::make_device_vector_view(r_1nn_cols2.data(), index.n_landmarks));
}

/**
 * Constructs a 1-nn index mapping each landmark to their closest points.
 * @tparam ValueIdx
 * @tparam ValueT
 * @param handle
 * @param R_knn_inds_ptr
 * @param R_knn_dists_ptr
 * @param k
 * @param index
 */
template <typename ValueIdx, typename ValueT>
void construct_landmark_1nn(raft::resources const& handle,
                            const ValueIdx* R_knn_inds_ptr,
                            const ValueT* R_knn_dists_ptr,
                            int64_t k,
                            cuvs::neighbors::ball_cover::index<ValueIdx, ValueT>& index)
{
  auto r_1nn_inds = raft::make_device_vector<ValueIdx, ValueIdx>(handle, index.m);

  thrust::fill(raft::resource::get_thrust_policy(handle),
               r_1nn_inds.data_handle(),
               r_1nn_inds.data_handle() + index.m,
               std::numeric_limits<ValueIdx>::max());

  raft::linalg::map_offset(
    handle, r_1nn_inds.view(), [R_knn_inds_ptr, k] __device__(ValueIdx i) -> ValueIdx {
      return R_knn_inds_ptr[i * k];
    });
  raft::linalg::map_offset(
    handle, index.get_R_1nn_dists(), [R_knn_dists_ptr, k] __device__(ValueIdx i) -> ValueT {
      return R_knn_dists_ptr[i * k];
    });

  auto keys = thrust::make_zip_iterator(
    thrust::make_tuple(r_1nn_inds.data_handle(), index.get_R_1nn_dists().data_handle()));

  // group neighborhoods for each reference landmark and sort each group by distance
  thrust::sort_by_key(raft::resource::get_thrust_policy(handle),
                      keys,
                      keys + index.m,
                      index.get_R_1nn_cols().data_handle(),
                      nn_comp());

  // convert to CSR for fast lookup
  raft::sparse::convert::sorted_coo_to_csr(r_1nn_inds.data_handle(),
                                           index.m,
                                           index.get_R_indptr().data_handle(),
                                           index.n_landmarks + 1,
                                           raft::resource::get_cuda_stream(handle));

  // reorder X to allow aligned access
  raft::matrix::copy_rows<ValueT, ValueIdx>(
    handle, index.get_X(), index.get_X_reordered(), index.get_R_1nn_cols());
}

/**
 * Computes the k closest landmarks to a set of query points.
 * @tparam ValueIdx
 * @tparam ValueT
 * @tparam value_int
 * @param handle
 * @param index
 * @param query_pts
 * @param n_query_pts
 * @param k
 * @param r_knn_inds
 * @param r_knn_dists
 */
template <typename ValueIdx, typename ValueT>
void k_closest_landmarks(raft::resources const& handle,
                         const cuvs::neighbors::ball_cover::index<ValueIdx, ValueT>& index,
                         const ValueT* query_pts,
                         int64_t n_query_pts,
                         int64_t k,
                         ValueIdx* r_knn_inds,
                         ValueT* r_knn_dists)
{
  raft::device_matrix_view<const ValueT, int64_t> inputs = index.get_R();
  auto bf_index_params   = cuvs::neighbors::brute_force::index_params();
  bf_index_params.metric = index.get_metric();
  auto bfknn             = cuvs::neighbors::brute_force::build(handle, bf_index_params, inputs);

  auto bf_search_params = cuvs::neighbors::brute_force::search_params();
  cuvs::neighbors::brute_force::search(
    handle,
    bf_search_params,
    bfknn,
    raft::make_device_matrix_view<const ValueT, int64_t>(query_pts, n_query_pts, inputs.extent(1)),
    raft::make_device_matrix_view<ValueIdx, int64_t>(r_knn_inds, n_query_pts, k),
    raft::make_device_matrix_view<ValueT, int64_t>(r_knn_dists, n_query_pts, k));
}

/**
 * Uses the sorted data points in the 1-nn landmark index to compute
 * an array of radii for each landmark.
 * @tparam ValueIdx
 * @tparam ValueT
 * @param handle
 * @param index
 */
template <typename ValueIdx, typename ValueT>
void compute_landmark_radii(raft::resources const& handle,
                            cuvs::neighbors::ball_cover::index<ValueIdx, ValueT>& index)
{
  const ValueIdx* r_indptr_ptr  = index.get_R_indptr().data_handle();
  const ValueT* r_1nn_dists_ptr = index.get_R_1nn_dists().data_handle();
  raft::linalg::map_offset(handle,
                           index.get_R_radius(),
                           [r_indptr_ptr, r_1nn_dists_ptr] __device__(ValueIdx input) -> ValueT {
                             ValueIdx last_row_idx = r_indptr_ptr[input + 1] - 1;
                             return r_1nn_dists_ptr[last_row_idx];
                           });
}

/**
 * 4. Perform k-select over original KNN, using L_r to filter distances
 *
 * a. Map 1 row to each warp/block
 * b. Add closest k R points to heap
 * c. Iterate through batches of R, having each thread in the warp load a set
 * of distances y from R (only if d(q, r) < 3 * distance to closest r) and
 * marking the distance to be computed between x, y only
 * if knn[k].distance >= d(x_i, R_k) + d(R_k, y)
 */
template <typename ValueIdx, typename ValueT>
void perform_rbc_query(raft::resources const& handle,
                       const cuvs::neighbors::ball_cover::index<ValueIdx, ValueT>& index,
                       const ValueT* query,
                       int64_t n_query_pts,
                       int64_t k,
                       const ValueIdx* r_knn_inds,
                       const ValueT* r_knn_dists,
                       ValueIdx* inds,
                       ValueT* dists,
                       float weight                = 1.0,
                       bool perform_post_filtering = true)
{
  // initialize output inds and dists
  thrust::fill(raft::resource::get_thrust_policy(handle),
               inds,
               inds + (k * n_query_pts),
               std::numeric_limits<ValueIdx>::max());
  thrust::fill(raft::resource::get_thrust_policy(handle),
               dists,
               dists + (k * n_query_pts),
               std::numeric_limits<ValueT>::max());

  // Compute nearest k for each neighborhood in each closest R
  rbc_low_dim_pass_one<ValueIdx, ValueT>(
    handle, index, query, n_query_pts, k, r_knn_inds, r_knn_dists, inds, dists, weight, index.n);

  if (perform_post_filtering) {
    rbc_low_dim_pass_two<ValueIdx, ValueT>(
      handle, index, query, n_query_pts, k, r_knn_inds, r_knn_dists, inds, dists, weight, index.n);
  }
}

/**
 * Perform eps-select
 *
 */
template <typename ValueIdx, typename ValueT, typename dist_func>
void perform_rbc_eps_nn_query(raft::resources const& handle,
                              const cuvs::neighbors::ball_cover::index<ValueIdx, ValueT>& index,
                              const ValueT* query,
                              int64_t n_query_pts,
                              ValueT eps,
                              const ValueT* landmarks,
                              dist_func dfunc,
                              bool* adj,
                              ValueIdx* vd)
{
  // initialize output
  RAFT_CUDA_TRY(cudaMemsetAsync(
    adj, 0, index.m * n_query_pts * sizeof(bool), raft::resource::get_cuda_stream(handle)));

  raft::resource::sync_stream(handle);

  rbc_eps_pass<ValueIdx, ValueT>(handle, index, query, n_query_pts, eps, landmarks, dfunc, adj, vd);

  raft::resource::sync_stream(handle);
}

template <typename ValueIdx, typename ValueT, typename dist_func>
void perform_rbc_eps_nn_query(raft::resources const& handle,
                              const cuvs::neighbors::ball_cover::index<ValueIdx, ValueT>& index,
                              const ValueT* query,
                              int64_t n_query_pts,
                              ValueT eps,
                              int64_t* max_k,
                              const ValueT* landmarks,
                              dist_func dfunc,
                              ValueIdx* adj_ia,
                              ValueIdx* adj_ja,
                              ValueIdx* vd)
{
  rbc_eps_pass<ValueIdx, ValueT>(
    handle, index, query, n_query_pts, eps, max_k, landmarks, dfunc, adj_ia, adj_ja, vd);

  raft::resource::sync_stream(handle);
}

/**
 * Similar to a ball tree, the random ball cover algorithm
 * uses the triangle inequality to prune distance computations
 * in any metric space with a guarantee of sqrt(n) * c^{3/2}
 * where `c` is an expansion constant based on the distance
 * metric.
 *
 * This function variant performs an all nearest neighbors
 * query which is useful for algorithms that need to perform
 * A * A.T.
 */
template <typename ValueIdx = std::int64_t, typename ValueT>
void rbc_build_index(raft::resources const& handle,
                     cuvs::neighbors::ball_cover::index<ValueIdx, ValueT>& index)
{
  ASSERT(!index.is_index_trained(), "index cannot be previously trained");

  rmm::device_uvector<ValueIdx> r_knn_inds(index.m, raft::resource::get_cuda_stream(handle));

  // Initialize the uvectors
  thrust::fill(raft::resource::get_thrust_policy(handle),
               r_knn_inds.begin(),
               r_knn_inds.end(),
               std::numeric_limits<ValueIdx>::max());
  thrust::fill(raft::resource::get_thrust_policy(handle),
               index.get_R_closest_landmark_dists().data_handle(),
               index.get_R_closest_landmark_dists().data_handle() + index.m,
               std::numeric_limits<ValueT>::max());

  /**
   * 1. Randomly sample sqrt(n) points from X
   */
  sample_landmarks<ValueIdx, ValueT>(handle, index);

  /**
   * 2. Perform knn = bfknn(X, R, k)
   */
  int64_t k = 1;
  k_closest_landmarks(handle,
                      index,
                      index.get_X().data_handle(),
                      index.m,
                      k,
                      r_knn_inds.data(),
                      index.get_R_closest_landmark_dists().data_handle());

  /**
   * 3. Create L_r = knn[:,0].T (CSR)
   *
   * Slice closest neighboring R
   * Secondary sort by (r_knn_inds, r_knn_dists)
   */
  construct_landmark_1nn(
    handle, r_knn_inds.data(), index.get_R_closest_landmark_dists().data_handle(), k, index);

  /**
   * Compute radius of each R for filtering: p(q, r) <= p(q, q_r) + radius(r)
   * (need to take the
   */
  compute_landmark_radii(handle, index);
}

/**
 * Performs an all neighbors knn query (e.g. index == query)
 */
template <typename ValueIdx = std::int64_t, typename ValueT>
void rbc_all_knn_query(raft::resources const& handle,
                       cuvs::neighbors::ball_cover::index<ValueIdx, ValueT>& index,
                       int64_t k,
                       ValueIdx* inds,
                       ValueT* dists,
                       // approximate nn options
                       bool perform_post_filtering = true,
                       float weight                = 1.0)
{
  ASSERT(index.n <= 3, "only 2d and 3d vectors are supported in current implementation");
  ASSERT(index.n_landmarks >= k, "number of landmark samples must be >= k");
  ASSERT(!index.is_index_trained(), "index cannot be previously trained");

  rmm::device_uvector<ValueIdx> r_knn_inds(k * index.m, raft::resource::get_cuda_stream(handle));
  rmm::device_uvector<ValueT> r_knn_dists(k * index.m, raft::resource::get_cuda_stream(handle));

  // Initialize the uvectors
  thrust::fill(raft::resource::get_thrust_policy(handle),
               r_knn_inds.begin(),
               r_knn_inds.end(),
               std::numeric_limits<ValueIdx>::max());
  thrust::fill(raft::resource::get_thrust_policy(handle),
               r_knn_dists.begin(),
               r_knn_dists.end(),
               std::numeric_limits<ValueT>::max());

  thrust::fill(raft::resource::get_thrust_policy(handle),
               inds,
               inds + (k * index.m),
               std::numeric_limits<ValueIdx>::max());
  thrust::fill(raft::resource::get_thrust_policy(handle),
               dists,
               dists + (k * index.m),
               std::numeric_limits<ValueT>::max());

  sample_landmarks<ValueIdx, ValueT>(handle, index);

  k_closest_landmarks(
    handle, index, index.get_X().data_handle(), index.m, k, r_knn_inds.data(), r_knn_dists.data());

  construct_landmark_1nn(handle, r_knn_inds.data(), r_knn_dists.data(), k, index);

  compute_landmark_radii(handle, index);

  perform_rbc_query(handle,
                    index,
                    index.get_X().data_handle(),
                    index.m,
                    k,
                    r_knn_inds.data(),
                    r_knn_dists.data(),
                    inds,
                    dists,
                    weight,
                    perform_post_filtering);
}

/**
 * Performs a knn query against an index. This assumes the index has
 * already been built.
 */
template <typename ValueIdx = std::int64_t, typename ValueT>
void rbc_knn_query(raft::resources const& handle,
                   const cuvs::neighbors::ball_cover::index<ValueIdx, ValueT>& index,
                   int64_t k,
                   const ValueT* query,
                   int64_t n_query_pts,
                   ValueIdx* inds,
                   ValueT* dists,
                   // approximate nn options
                   bool perform_post_filtering = true,
                   float weight                = 1.0)
{
  ASSERT(index.n <= 3, "only 2d and 3d vectors are supported in current implementation");
  ASSERT(index.n_landmarks >= k, "number of landmark samples must be >= k");
  ASSERT(index.is_index_trained(), "index must be previously trained");

  rmm::device_uvector<ValueIdx> r_knn_inds(k * n_query_pts,
                                           raft::resource::get_cuda_stream(handle));
  rmm::device_uvector<ValueT> r_knn_dists(k * n_query_pts, raft::resource::get_cuda_stream(handle));

  // Initialize the uvectors
  thrust::fill(raft::resource::get_thrust_policy(handle),
               r_knn_inds.begin(),
               r_knn_inds.end(),
               std::numeric_limits<ValueIdx>::max());
  thrust::fill(raft::resource::get_thrust_policy(handle),
               r_knn_dists.begin(),
               r_knn_dists.end(),
               std::numeric_limits<ValueT>::max());

  thrust::fill(raft::resource::get_thrust_policy(handle),
               inds,
               inds + (k * n_query_pts),
               std::numeric_limits<ValueIdx>::max());
  thrust::fill(raft::resource::get_thrust_policy(handle),
               dists,
               dists + (k * n_query_pts),
               std::numeric_limits<ValueT>::max());

  k_closest_landmarks(handle, index, query, n_query_pts, k, r_knn_inds.data(), r_knn_dists.data());

  perform_rbc_query(handle,
                    index,
                    query,
                    n_query_pts,
                    k,
                    r_knn_inds.data(),
                    r_knn_dists.data(),
                    inds,
                    dists,
                    weight,
                    perform_post_filtering);
}

template <typename ValueIdx, typename ValueT>
void compute_landmark_dists(raft::resources const& handle,
                            const cuvs::neighbors::ball_cover::index<ValueIdx, ValueT>& index,
                            const ValueT* query_pts,
                            int64_t n_query_pts,
                            ValueT* R_dists)
{
  // compute distances for all queries against all landmarks
  // index.get_R() -- landmark points in row order (index.n_landmarks x index.k)
  // query_pts     -- query points in row order (n_query_pts x index.k)
  RAFT_EXPECTS(std::max<size_t>(index.n_landmarks, n_query_pts) * index.n <
                 static_cast<size_t>(std::numeric_limits<int>::max()),
               "Too large input for pairwise_distance with `int` index.");
  RAFT_EXPECTS(n_query_pts * static_cast<size_t>(index.n_landmarks) <
                 static_cast<size_t>(std::numeric_limits<int>::max()),
               "Too large input for pairwise_distance with `int` index.");
  cuvs::distance::pairwise_distance(handle,
                                    query_pts,
                                    index.get_R().data_handle(),
                                    R_dists,
                                    n_query_pts,
                                    index.n_landmarks,
                                    index.n,
                                    index.get_metric());
}

/**
 * Performs a knn query against an index. This assumes the index has
 * already been built.
 * Modified version that takes an eps as threshold and outputs to a dense adj matrix (row-major)
 * we are assuming that there are sufficiently many landmarks
 */
template <typename ValueIdx = std::int64_t, typename ValueT, typename DistanceFunc>
void rbc_eps_nn_query(raft::resources const& handle,
                      const cuvs::neighbors::ball_cover::index<ValueIdx, ValueT>& index,
                      const ValueT eps,
                      const ValueT* query,
                      int64_t n_query_pts,
                      bool* adj,
                      ValueIdx* vd,
                      DistanceFunc dfunc)
{
  ASSERT(index.is_index_trained(), "index must be previously trained");

  // query all points and write to adj
  perform_rbc_eps_nn_query(
    handle, index, query, n_query_pts, eps, index.get_R().data_handle(), dfunc, adj, vd);
}

template <typename ValueIdx = std::int64_t, typename ValueT, typename DistanceFunc>
void rbc_eps_nn_query(raft::resources const& handle,
                      const cuvs::neighbors::ball_cover::index<ValueIdx, ValueT>& index,
                      const ValueT eps,
                      int64_t* max_k,
                      const ValueT* query,
                      int64_t n_query_pts,
                      ValueIdx* adj_ia,
                      ValueIdx* adj_ja,
                      ValueIdx* vd,
                      DistanceFunc dfunc)
{
  ASSERT(index.is_index_trained(), "index must be previously trained");

  // query all points and write to adj
  perform_rbc_eps_nn_query(handle,
                           index,
                           query,
                           n_query_pts,
                           eps,
                           max_k,
                           index.get_R().data_handle(),
                           dfunc,
                           adj_ia,
                           adj_ja,
                           vd);
}

};  // namespace cuvs::neighbors::ball_cover::detail
