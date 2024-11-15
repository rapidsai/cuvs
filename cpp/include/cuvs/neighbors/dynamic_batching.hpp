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
  /** Filtering function, if any, must be the same for all requests in a batch. */
  cuvs::neighbors::filtering::base_filter* sample_filter = nullptr;
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
  /**
   * By default (`conservative_dispatch = false`) the first CPU thread to commit a query to a batch
   * dispatches the upstream search function as soon as possible (before the batch is full). In that
   * case, it does not know the final batch size at the time of calling the upstream search and thus
   * runs the upstream search with the maximum batch size every time, even if only one valid query
   * is present in the batch. This reduces the latency at the cost of wasted GPU resources.
   *
   * The alternative behavaior (`conservative_dispatch = true`) is more conservative: the dispatcher
   * thread starts the kernel that gathers input queries, but waits till the batch is full or the
   * waiting time is exceeded. Only then it acquires the actual batch size and launches the upstream
   * search. As a result, no GPU resources are wasted at the cost of exposed upstream search
   * latency.
   *
   * Rule of Thumb:
   *    for a large `max_batch_size` set `conservative_dispatch = true`, otherwise keep it disabled.
   */
  bool conservative_dispatch = false;
};

struct search_params : cuvs::neighbors::search_params {
  /**
   * How long a request can stay in the queue (milliseconds).
   * Note, this only affects the dispatch time and does not reflect full request latency;
   * the latter depends on the upstream search parameters and the batch size.
   */
  double dispatch_timeout_ms = 1.0;
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
            dynamic_batching::index<half, uint32_t> const& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
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
            dynamic_batching::index<half, int64_t> const& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
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
