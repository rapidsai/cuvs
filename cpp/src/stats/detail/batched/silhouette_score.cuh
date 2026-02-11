/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

namespace cuvs::stats::batched::detail {

/**
 * This kernel initializes matrix b (n_rows * n_labels)
 * For each label that the corresponding row is not a part of is initialized as 0
 * If the corresponding row is the only sample in its label, again 0
 * Only if the there are > 1 samples in the label, row is initialized to max
 */
template <typename ValueT, typename ValueIdx, typename LabelIdx>
RAFT_KERNEL fill_b_kernel(
  ValueT* b, const LabelIdx* y, ValueIdx n_rows, LabelIdx n_labels, const ValueIdx* cluster_counts)
{
  ValueIdx idx = threadIdx.x + blockIdx.x * blockDim.x;
  LabelIdx idy = threadIdx.y + blockIdx.y * blockDim.y;

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
      b[idx * n_labels + idy] = std::numeric_limits<ValueT>::max();
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
template <typename ValueT, typename ValueIdx, typename LabelIdx>
RAFT_KERNEL compute_chunked_a_b_kernel(ValueT* a,
                                       ValueT* b,
                                       ValueIdx row_offset,
                                       ValueIdx col_offset,
                                       const LabelIdx* y,
                                       LabelIdx n_labels,
                                       const ValueIdx* cluster_counts,
                                       const ValueT* distances,
                                       ValueIdx dist_rows,
                                       ValueIdx dist_cols)
{
  ValueIdx row_id = threadIdx.x + blockIdx.x * blockDim.x;
  ValueIdx col_id = threadIdx.y + blockIdx.y * blockDim.y;

  // these are global offsets of current element
  // in the full pairwise distance matrix
  ValueIdx pw_row_id = row_id + row_offset;
  ValueIdx pw_col_id = col_id + col_offset;

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

template <typename ValueIdx, typename LabelIdx>
auto get_cluster_counts(raft::resources const& handle,
                        const LabelIdx* y,
                        ValueIdx& n_rows,
                        LabelIdx& n_labels) -> rmm::device_uvector<ValueIdx>
{
  auto stream = raft::resource::get_cuda_stream(handle);

  rmm::device_uvector<ValueIdx> cluster_counts(n_labels, stream);

  rmm::device_uvector<char> workspace(1, stream);

  cuvs::stats::detail::count_labels(y, cluster_counts.data(), n_rows, n_labels, workspace, stream);

  return cluster_counts;
}

template <typename ValueT, typename ValueIdx>
auto get_pairwise_distance(raft::resources const& handle,
                           const ValueT* left_begin,
                           const ValueT* right_begin,
                           ValueIdx& n_left_rows,
                           ValueIdx& n_right_rows,
                           ValueIdx& n_cols,
                           cuvs::distance::DistanceType metric,
                           cudaStream_t stream) -> rmm::device_uvector<ValueT>
{
  rmm::device_uvector<ValueT> distances(n_left_rows * n_right_rows, stream);

  cuvs::distance::pairwise_distance(
    handle,
    raft::make_device_matrix_view<const ValueT, int64_t>(left_begin, n_left_rows, n_cols),
    raft::make_device_matrix_view<const ValueT, int64_t>(right_begin, n_right_rows, n_cols),
    raft::make_device_matrix_view<ValueT, int64_t>(distances.data(), n_left_rows, n_right_rows),
    metric);

  return distances;
}

template <typename ValueT, typename ValueIdx, typename LabelIdx>
void compute_chunked_a_b(raft::resources const& handle,
                         ValueT* a,
                         ValueT* b,
                         ValueIdx& row_offset,
                         ValueIdx& col_offset,
                         const LabelIdx* y,
                         LabelIdx& n_labels,
                         const ValueIdx* cluster_counts,
                         const ValueT* distances,
                         ValueIdx& dist_rows,
                         ValueIdx& dist_cols,
                         cudaStream_t stream)
{
  dim3 block_size(std::min(dist_rows, 32), std::min(dist_cols, 32));
  dim3 grid_size(raft::ceildiv(dist_rows, static_cast<ValueIdx>(block_size.x)),
                 raft::ceildiv(dist_cols, static_cast<ValueIdx>(block_size.y)));

  detail::compute_chunked_a_b_kernel<<<grid_size, block_size, 0, stream>>>(
    a, b, row_offset, col_offset, y, n_labels, cluster_counts, distances, dist_rows, dist_cols);
}

template <typename ValueT, typename ValueIdx, typename LabelIdx>
auto silhouette_score(
  raft::resources const& handle,
  const ValueT* X,
  ValueIdx n_rows,
  ValueIdx n_cols,
  const LabelIdx* y,
  LabelIdx n_labels,
  ValueT* scores,
  ValueIdx chunk,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded) -> ValueT
{
  ASSERT(n_labels >= 2 && n_labels <= (n_rows - 1),
         "silhouette Score not defined for the given number of labels!");

  rmm::device_uvector<ValueIdx> cluster_counts = get_cluster_counts(handle, y, n_rows, n_labels);

  auto stream = raft::resource::get_cuda_stream(handle);
  auto policy = raft::resource::get_thrust_policy(handle);

  auto b_size = n_rows * n_labels;

  ValueT *a_ptr, *b_ptr;
  rmm::device_uvector<ValueT> a(0, stream);
  rmm::device_uvector<ValueT> b(b_size, stream);

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
  dim3 grid_size(raft::ceildiv(n_rows, static_cast<ValueIdx>(block_size.x)),
                 raft::ceildiv(n_labels, static_cast<LabelIdx>(block_size.y)));
  detail::fill_b_kernel<<<grid_size, block_size, 0, stream>>>(
    b_ptr, y, n_rows, n_labels, cluster_counts.data());

  raft::resource::wait_stream_pool_on_stream(handle);

  auto n_iters = 0;

  for (ValueIdx i = 0; i < n_rows; i += chunk) {
    for (ValueIdx j = 0; j < n_rows; j += chunk) {
      ++n_iters;

      auto chunk_stream = raft::resource::get_next_usable_stream(handle, i + chunk * j);

      const auto* left_begin  = X + (i * n_cols);
      const auto* right_begin = X + (j * n_cols);

      auto n_left_rows  = (i + chunk) < n_rows ? chunk : (n_rows - i);
      auto n_right_rows = (j + chunk) < n_rows ? chunk : (n_rows - j);

      rmm::device_uvector<ValueT> distances = get_pairwise_distance(
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
  raft::linalg::reduce<true, true, ValueT, ValueT, ValueIdx, raft::identity_op, raft::min_op>(
    b_ptr,
    b_ptr,
    n_labels,
    n_rows,
    std::numeric_limits<ValueT>::max(),
    stream,
    false,
    raft::identity_op(),
    raft::min_op());

  // calculating the silhouette score per sample
  raft::linalg::binaryOp<ValueT, cuvs::stats::detail::sil_op<ValueT>, ValueT, ValueIdx>(
    a_ptr, a_ptr, b_ptr, n_rows, cuvs::stats::detail::sil_op<ValueT>(), stream);

  return thrust::reduce(policy, a_ptr, a_ptr + n_rows, ValueT(0)) / n_rows;
}

}  // namespace cuvs::stats::batched::detail
