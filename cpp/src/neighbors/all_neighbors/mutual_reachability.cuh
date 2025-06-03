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
#include <raft/core/detail/macros.hpp>
#include <raft/matrix/shift.cuh>
#include <rmm/exec_policy.hpp>

namespace cuvs::neighbors {

//  Functor to post-process distances into reachability space
template <typename value_idx, typename value_t>
struct ReachabilityPostProcess {
  RAFT_DEVICE_INLINE_FUNCTION value_t operator()(value_t value, value_idx row, value_idx col) const
  {
    return max(core_dists[col], max(core_dists[row], alpha * value));
  }

  const value_t* core_dists;
  value_t alpha;
  size_t n;  // total number of elements
};

template <typename value_idx, typename value_t>
void core_distances(
  value_t* knn_dists, int min_samples, int n_neighbors, size_t n, value_t* out, cudaStream_t stream)
{
  ASSERT(n_neighbors >= min_samples,
         "the size of the neighborhood should be greater than or equal to min_samples");

  auto exec_policy = rmm::exec_policy(stream);

  auto indices = thrust::make_counting_iterator<value_idx>(0);

  thrust::transform(exec_policy, indices, indices + n, out, [=] __device__(value_idx row) {
    return knn_dists[row * n_neighbors + (min_samples - 1)];
  });
}

template <typename value_idx, typename value_t>
void get_core_distances(const raft::resources& handle,
                        raft::device_matrix_view<value_t, value_idx> knn_dists,
                        raft::device_vector_view<value_t, value_idx> core_dists,
                        bool need_shift = false)
{
  size_t num_rows = static_cast<size_t>(knn_dists.extent(0));
  size_t k        = static_cast<size_t>(knn_dists.extent(1));

  if (need_shift) {
    raft::matrix::shift(handle, knn_dists, 1, std::make_optional(static_cast<value_t>(0.0)));
  }

  core_distances<value_idx, value_t>(knn_dists.data_handle(),
                                     k,
                                     k,
                                     num_rows,
                                     core_dists.data_handle(),
                                     raft::resource::get_cuda_stream(handle));
}
}  // namespace cuvs::neighbors
