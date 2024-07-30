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

#include <cuvs/neighbors/common.hpp>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

namespace cuvs::neighbors::dynamic_batching {

namespace detail {
template <typename T, typename IdxT>
class batch_runner;
}

template <typename Upstream>
struct index_params : cuvs::neighbors::index_params {
  /** The index, to which the dynamic batches are dispatched. */
  Upstream& upstream;
  /** Search paramerets for all requests within a batch. */
  typename Upstream::search_params_type& upstream_params;
  /** The number of neighbors to search is fixed at construction time. */
  int64_t k;
  /** Input data (queries) dimensionality. */
  int64_t dim;
  /** Maximum size of the batch to submit to the upstream index. */
  int64_t max_batch_size = 100;
  /**
   * The number of independent request queues.
   *
   * Each queue is associated with a unique CUDA stream and IO device buffers. If the number of
   * concurrent requests is high, using multiple queues allows to fill-in data and prepare the batch
   * while the other queue is busy. Moreover, the queues are submitted concurrently; this allows to
   * better utilize the GPU by hiding the the kernel launch latencies, which helps to improve the
   * throughput.
   */
  size_t n_queues = 3;
};

struct search_params : cuvs::neighbors::search_params {
  /** How long a request can stay in the queue (milliseconds). */
  double soft_deadline_ms = 1.0;
};

/**
 * @brief Lightweight dynamic batching index wrapper
 *
 * One lightweight dynamic batching index manages a single index and a single search parameter set.
 * Upon creation, it starts a separate thread to serve the incoming requests.
 * The server thread is terminated on the destruction of the index.
 */
template <typename T, typename IdxT>
struct index : cuvs::neighbors::index {
  std::shared_ptr<detail::batch_runner<T, IdxT>> runner;

  template <typename Upstream>
  index(const raft::resources& res, const dynamic_batching::index_params<Upstream>& params);
};

void search(raft::resources const& res,
            cuvs::neighbors::dynamic_batching::search_params const& params,
            dynamic_batching::index<float, uint32_t> const& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

void search(raft::resources const& res,
            cuvs::neighbors::dynamic_batching::search_params const& params,
            dynamic_batching::index<int8_t, uint32_t> const& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

void search(raft::resources const& res,
            cuvs::neighbors::dynamic_batching::search_params const& params,
            dynamic_batching::index<uint8_t, uint32_t> const& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

void search(raft::resources const& res,
            cuvs::neighbors::dynamic_batching::search_params const& params,
            dynamic_batching::index<float, int64_t> const& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

void search(raft::resources const& res,
            cuvs::neighbors::dynamic_batching::search_params const& params,
            dynamic_batching::index<int8_t, int64_t> const& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

void search(raft::resources const& res,
            cuvs::neighbors::dynamic_batching::search_params const& params,
            dynamic_batching::index<uint8_t, int64_t> const& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

}  // namespace cuvs::neighbors::dynamic_batching
