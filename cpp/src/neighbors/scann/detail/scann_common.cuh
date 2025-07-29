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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/matrix/gather.cuh>

namespace cuvs::neighbors::experimental::scann::detail {

template <typename T, typename IdxT>
struct gather_functor {
  auto operator()(raft::resources const& res,
                  raft::device_matrix_view<const T, int64_t> dataset,
                  raft::device_vector_view<const IdxT, int64_t> cluster_ids,
                  raft::device_matrix_view<T, int64_t> cluster_vecs,
                  cudaStream_t stream)
  {
    raft::matrix::gather(
      res, raft::make_const_mdspan(dataset), raft::make_const_mdspan(cluster_ids), cluster_vecs);
  }

  auto operator()(raft::resources const& res,
                  raft::host_matrix_view<const T, int64_t> dataset,
                  raft::device_vector_view<const IdxT, int64_t> cluster_ids,
                  raft::device_matrix_view<T, int64_t> cluster_vecs,
                  cudaStream_t stream)
  {
    auto h_cluster_ids = raft::make_host_vector<IdxT, int64_t>(cluster_ids.extent(0));

    raft::copy(h_cluster_ids.data_handle(), cluster_ids.data_handle(), cluster_ids.size(), stream);

    auto pinned_cluster =
      raft::make_host_matrix<T, int64_t>(cluster_vecs.extent(0), cluster_vecs.extent(1));

    raft::resource::sync_stream(res, stream);

    int n_threads = std::min<int>(omp_get_max_threads(), 32);

#pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < h_cluster_ids.extent(0); i++) {
      // std::
      memcpy(pinned_cluster.data_handle() + i * pinned_cluster.extent(1),
             dataset.data_handle() + h_cluster_ids(i) * dataset.extent(1),
             sizeof(T) * dataset.extent(1));
    }

    raft::copy(
      cluster_vecs.data_handle(), pinned_cluster.data_handle(), pinned_cluster.size(), stream);
    raft::resource::sync_stream(res, stream);
  }
};

}  // namespace cuvs::neighbors::experimental::scann::detail
