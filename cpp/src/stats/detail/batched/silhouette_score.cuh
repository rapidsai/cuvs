/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "../silhouette_score.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/device_atomics.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

namespace cuvs {
namespace stats {
namespace batched {
namespace detail {

/**
 * This kernel initializes matrix b (n_rows * n_labels)
 * For each label that the corresponding row is not a part of is initialized as 0
 * If the corresponding row is the only sample in its label, again 0
 * Only if the there are > 1 samples in the label, row is initialized to max
 */
template <typename value_t, typename value_idx, typename label_idx>
RAFT_KERNEL fill_b_kernel(value_t* b,
                          const label_idx* y,
                          value_idx n_rows,
                          label_idx n_labels,
                          const value_idx* cluster_counts)
{
  value_idx idx = threadIdx.x + blockIdx.x * blockDim.x;
  label_idx idy = threadIdx.y + blockIdx.y * blockDim.y;

  if (idx >= n_rows || idy >= n_labels) { return; }

  auto row_cluster = y[idx];

  auto col_cluster_count = cluster_counts[idy];

  // b for own cluster should be max value
  // so that it does not interfere with min operator
  // b is also max if col cluster count is 0
  // however, b is 0 if self cluster count is 1
  if (row_cluster == idy || col_cluster_count == 0) {
    if (cluster_counts[row_cluster] == 1) {
      b[idx * n_labels + idy] = 0;
    } else {
      b[idx * n_labels + idy] = std::numeric_limits<value_t>::max();
    }
  } else {
    b[idx * n_labels + idy] = 0;
  }
}

/**
 * This kernel does an elementwise sweep of chunked pairwise distance matrix
 * By knowing the offsets of the chunked pairwise distance matrix in the
 * global pairwise distance matrix, we are able to calculate
 * intermediate values of a and b for the rows and columns present in the
 * current chunked pairwise distance matrix.
 */
template <typename value_t, typename value_idx, typename label_idx>
RAFT_KERNEL compute_chunked_a_b_kernel(value_t* a,
                                       value_t* b,
                                       value_idx row_offset,
                                       value_idx col_offset,
                                       const label_idx* y,
                                       label_idx n_labels,
                                       const value_idx* cluster_counts,
                                       const value_t* distances,
                                       value_idx dist_rows,
                                       value_idx dist_cols)
{
  value_idx row_id = threadIdx.x + blockIdx.x * blockDim.x;
  value_idx col_id = threadIdx.y + blockIdx.y * blockDim.y;

  // these are global offsets of current element
  // in the full pairwise distance matrix
  value_idx pw_row_id = row_id + row_offset;
  value_idx pw_col_id = col_id + col_offset;

  if (row_id >= dist_rows || col_id >= dist_cols || pw_row_id == pw_col_id) { return; }

  auto row_cluster = y[pw_row_id];
  if (cluster_counts[row_cluster] == 1) { return; }

  auto col_cluster        = y[pw_col_id];
  auto col_cluster_counts = cluster_counts[col_cluster];

  if (col_cluster == row_cluster) {
    atomicAdd(&a[pw_row_id], distances[row_id * dist_cols + col_id] / (col_cluster_counts - 1));
  } else {
    atomicAdd(&b[pw_row_id * n_labels + col_cluster],
              distances[row_id * dist_cols + col_id] / col_cluster_counts);
  }
}

template <typename value_idx, typename label_idx>
rmm::device_uvector<value_idx> get_cluster_counts(raft::resources const& handle,
                                                  const label_idx* y,
                                                  value_idx& n_rows,
                                                  label_idx& n_labels)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  rmm::device_uvector<value_idx> cluster_counts(n_labels, stream);

  rmm::device_uvector<char> workspace(1, stream);

  cuvs::stats::detail::countLabels(y, cluster_counts.data(), n_rows, n_labels, workspace, stream);

  return cluster_counts;
}

template <typename value_t, typename value_idx>
rmm::device_uvector<value_t> get_pairwise_distance(raft::resources const& handle,
                                                   const value_t* left_begin,
                                                   const value_t* right_begin,
                                                   value_idx& n_left_rows,
                                                   value_idx& n_right_rows,
                                                   value_idx& n_cols,
                                                   cuvs::distance::DistanceType metric,
                                                   cudaStream_t stream)
{
  rmm::device_uvector<value_t> distances(n_left_rows * n_right_rows, stream);

  cuvs::distance::pairwise_distance(
    handle,
    raft::make_device_matrix_view<const value_t, int64_t>(left_begin, n_left_rows, n_cols),
    raft::make_device_matrix_view<const value_t, int64_t>(right_begin, n_right_rows, n_cols),
    raft::make_device_matrix_view<value_t, int64_t>(distances.data(), n_left_rows, n_right_rows),
    metric);

  return distances;
}

template <typename value_t, typename value_idx, typename label_idx>
void compute_chunked_a_b(raft::resources const& handle,
                         value_t* a,
                         value_t* b,
                         value_idx& row_offset,
                         value_idx& col_offset,
                         const label_idx* y,
                         label_idx& n_labels,
                         const value_idx* cluster_counts,
                         const value_t* distances,
                         value_idx& dist_rows,
                         value_idx& dist_cols,
                         cudaStream_t stream)
{
  dim3 block_size(std::min(dist_rows, 32), std::min(dist_cols, 32));
  dim3 grid_size(raft::ceildiv(dist_rows, (value_idx)block_size.x),
                 raft::ceildiv(dist_cols, (value_idx)block_size.y));

  detail::compute_chunked_a_b_kernel<<<grid_size, block_size, 0, stream>>>(
    a, b, row_offset, col_offset, y, n_labels, cluster_counts, distances, dist_rows, dist_cols);
}

template <typename value_t, typename value_idx, typename label_idx>
value_t silhouette_score(
  raft::resources const& handle,
  const value_t* X,
  value_idx n_rows,
  value_idx n_cols,
  const label_idx* y,
  label_idx n_labels,
  value_t* scores,
  value_idx chunk,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded)
{
  ASSERT(n_labels >= 2 && n_labels <= (n_rows - 1),
         "silhouette Score not defined for the given number of labels!");

  rmm::device_uvector<value_idx> cluster_counts = get_cluster_counts(handle, y, n_rows, n_labels);

  auto stream = raft::resource::get_cuda_stream(handle);
  auto policy = raft::resource::get_thrust_policy(handle);

  auto b_size = n_rows * n_labels;

  value_t *a_ptr, *b_ptr;
  rmm::device_uvector<value_t> a(0, stream);
  rmm::device_uvector<value_t> b(b_size, stream);

  b_ptr = b.data();

  // since a and silhouette score per sample are same size, reusing
  if (scores == nullptr || scores == NULL) {
    a.resize(n_rows, stream);
    a_ptr = a.data();
  } else {
    a_ptr = scores;
  }

  thrust::fill(policy, a_ptr, a_ptr + n_rows, 0);

  dim3 block_size(std::min(n_rows, 32), std::min(n_labels, 32));
  dim3 grid_size(raft::ceildiv(n_rows, (value_idx)block_size.x),
                 raft::ceildiv(n_labels, (label_idx)block_size.y));
  detail::fill_b_kernel<<<grid_size, block_size, 0, stream>>>(
    b_ptr, y, n_rows, n_labels, cluster_counts.data());

  raft::resource::wait_stream_pool_on_stream(handle);

  auto n_iters = 0;

  for (value_idx i = 0; i < n_rows; i += chunk) {
    for (value_idx j = 0; j < n_rows; j += chunk) {
      ++n_iters;

      auto chunk_stream = raft::resource::get_next_usable_stream(handle, i + chunk * j);

      const auto* left_begin  = X + (i * n_cols);
      const auto* right_begin = X + (j * n_cols);

      auto n_left_rows  = (i + chunk) < n_rows ? chunk : (n_rows - i);
      auto n_right_rows = (j + chunk) < n_rows ? chunk : (n_rows - j);

      rmm::device_uvector<value_t> distances = get_pairwise_distance(
        handle, left_begin, right_begin, n_left_rows, n_right_rows, n_cols, metric, chunk_stream);

      compute_chunked_a_b(handle,
                          a_ptr,
                          b_ptr,
                          i,
                          j,
                          y,
                          n_labels,
                          cluster_counts.data(),
                          distances.data(),
                          n_left_rows,
                          n_right_rows,
                          chunk_stream);
    }
  }

  raft::resource::sync_stream_pool(handle);

  // calculating row-wise minimum in b
  // this prim only supports int indices for now
  raft::linalg::reduce<true, true, value_t, value_t, value_idx, raft::identity_op, raft::min_op>(
    b_ptr,
    b_ptr,
    n_labels,
    n_rows,
    std::numeric_limits<value_t>::max(),
    stream,
    false,
    raft::identity_op(),
    raft::min_op());

  // calculating the silhouette score per sample
  raft::linalg::binaryOp<value_t, cuvs::stats::detail::SilOp<value_t>, value_t, value_idx>(
    a_ptr, a_ptr, b_ptr, n_rows, cuvs::stats::detail::SilOp<value_t>(), stream);

  return thrust::reduce(policy, a_ptr, a_ptr + n_rows, value_t(0)) / n_rows;
}

}  // namespace detail
}  // namespace batched
}  // namespace stats
}  // namespace cuvs
