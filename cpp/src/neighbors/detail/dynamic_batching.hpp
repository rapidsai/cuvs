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

#include <cuvs/neighbors/dynamic_batching.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>

#include <cuda/std/atomic>
#include <rmm/resource_ref.hpp>

#include <mutex>
#include <queue>

namespace cuvs::neighbors::dynamic_batching::detail {

constexpr size_t kCacheLineBytes = 64;

template <typename Upstream, typename T, typename IdxT>
using upstream_search_type_const = void(raft::resources const&,
                                        typename Upstream::search_params_type const&,
                                        Upstream const&,
                                        raft::device_matrix_view<const T, int64_t, raft::row_major>,
                                        raft::device_matrix_view<IdxT, int64_t, raft::row_major>,
                                        raft::device_matrix_view<float, int64_t, raft::row_major>);

template <typename Upstream, typename T, typename IdxT>
using upstream_search_type = void(raft::resources const&,
                                  typename Upstream::search_params_type const&,
                                  Upstream&,
                                  raft::device_matrix_view<const T, int64_t, raft::row_major>,
                                  raft::device_matrix_view<IdxT, int64_t, raft::row_major>,
                                  raft::device_matrix_view<float, int64_t, raft::row_major>);

template <typename T, typename IdxT>
using function_search_type = void(raft::resources const&,
                                  raft::device_matrix_view<const T, int64_t, raft::row_major>,
                                  raft::device_matrix_view<IdxT, int64_t, raft::row_major>,
                                  raft::device_matrix_view<float, int64_t, raft::row_major>);

template <typename T, typename IdxT>
struct alignas(kCacheLineBytes) batch {
  using time_point = std::chrono::time_point<std::chrono::system_clock>;
  static_assert(cuda::std::atomic<int64_t>::is_always_lock_free);
  static_assert(cuda::std::atomic<time_point>::is_always_lock_free);

  std::mutex lock;
  raft::resources res;
  cuda::std::atomic<int64_t> size{0};
  cuda::std::atomic<time_point> deadline{time_point::max()};
  cudaStream_t stream = nullptr;
  cudaEvent_t event   = nullptr;
  raft::device_matrix<T, int64_t, raft::row_major> queries;
  raft::device_matrix<IdxT, int64_t, raft::row_major> neighbors;
  raft::device_matrix<float, int64_t, raft::row_major> distances;

  batch(const raft::resources& res_from_runner,
        rmm::device_async_resource_ref mr,
        int64_t batch_size,
        int64_t dim,
        int64_t k)
    : res{res_from_runner},
      queries{raft::make_device_mdarray<T>(
        res_from_runner, mr, raft::make_extents<int64_t>(batch_size, dim))},
      neighbors{raft::make_device_mdarray<IdxT>(
        res_from_runner, mr, raft::make_extents<int64_t>(batch_size, k))},
      distances{raft::make_device_mdarray<float>(
        res_from_runner, mr, raft::make_extents<int64_t>(batch_size, k))}
  {
    RAFT_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    RAFT_CUDA_TRY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    raft::resource::set_cuda_stream(res, stream);
  }

  batch(const raft::resources& res, int64_t batch_size, int64_t dim, int64_t k)
    : batch{res, raft::resource::get_workspace_resource(res), batch_size, dim, k}
  {
  }

  batch(batch<T, IdxT>&&)                               = delete;
  auto operator=(batch<T, IdxT>&&) -> batch&            = delete;
  batch(const batch<T, IdxT>& res)                      = delete;
  auto operator=(const batch<T, IdxT>& other) -> batch& = delete;

  ~batch() noexcept
  {
    RAFT_CUDA_TRY_NO_THROW(cudaStreamDestroy(stream));
    RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(event));
  }

  [[nodiscard]] auto capacity() const -> int64_t { return queries.extent(0); }
};

template <typename T, typename IdxT>
class batch_runner {
 public:
  // Save the parameters and the upstream batched search function to invoke
  template <typename Upstream>
  batch_runner(const raft::resources& res,
               const dynamic_batching::index_params<Upstream>& params,
               const upstream_search_type_const<Upstream, T, IdxT>* upstream_search)
    : res_(res),
      upstream_search_{[&ix = params.upstream, &ps = params.upstream_params, upstream_search](
                         raft::resources const& res,
                         raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
                         raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,
                         raft::device_matrix_view<float, int64_t, raft::row_major> distances) {
        return upstream_search(res, ps, ix, queries, neighbors, distances);
      }},
      n_queues_{params.n_queues},
      batch_bufs_{reinterpret_cast<batch<T, IdxT>*>(
        aligned_alloc(alignof(batch<T, IdxT>), n_queues_ * sizeof(batch<T, IdxT>)))}
  {
    for (size_t i = 0; i < n_queues_; i++) {
      new (batch_bufs_ + i) batch<T, IdxT>(res_, params.max_batch_size, params.dim, params.k);
    }
  }

  // A workaround for algos, which have non-const `index` type in their arguments
  template <typename Upstream>
  batch_runner(const raft::resources& res,
               const dynamic_batching::index_params<Upstream>& params,
               const upstream_search_type<Upstream, T, IdxT>* upstream_search)
    : batch_runner{
        res,
        params,
        reinterpret_cast<const upstream_search_type_const<Upstream, T, IdxT>*>(upstream_search)}
  {
  }

  ~batch_runner() noexcept
  {
    for (size_t i = 0; i < n_queues_; i++) {
      batch_bufs_[i].~batch();
    }
    free(batch_bufs_);
  }

  void search(raft::resources const& res,
              cuvs::neighbors::dynamic_batching::search_params const& params,
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
              raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,
              raft::device_matrix_view<float, int64_t, raft::row_major> distances) const
  {
    auto user_stream = raft::resource::get_cuda_stream(res);
    auto deadline    = std::chrono::system_clock::now() +
                    std::chrono::nanoseconds(size_t(params.soft_deadline_ms * 1000000.0));
    int64_t submitted_queries     = queries.extent(0);
    int64_t my_submit_counter     = submit_counter.load();
    int64_t batch_offset          = 0;
    batch<T, IdxT>* current_batch = nullptr;
    auto check_counter            = [&global_counter = submit_counter, &my_submit_counter]() {
      auto x = global_counter.load();
      if (x == my_submit_counter) { return true; }
      my_submit_counter = x;
      return false;
    };
    while (true) {
      current_batch = &batch_bufs_[my_submit_counter % n_queues_];
      std::lock_guard<std::mutex> guard(current_batch->lock);
      if (!check_counter()) { continue; }
      batch_offset      = current_batch->size.load();
      auto free_spots   = current_batch->capacity() - batch_offset;
      submitted_queries = std::min(submitted_queries, free_spots);
      raft::copy(current_batch->queries.data_handle() + batch_offset * queries.extent(1),
                 queries.data_handle(),
                 submitted_queries * queries.extent(1),
                 current_batch->stream);
      current_batch->size += submitted_queries;
      deadline = std::min(current_batch->deadline.load(), deadline);
      current_batch->deadline.store(deadline);
      /* [ Note: events and streams ]
      The queries are copied to the batch in the user stream, then the batch event is recorded.
      If the batch is full by this point, that means other users have already been through this
      section: they initiated async copies in their streams and recorded the event.
      Therefore, the event has captured the content of all participating user streams.
      Hence, we can safely submit the batch in the user stream - if it waits on the event first that
      is.

      The `submit` function first waits on the event, then runs the search, then records the event
      again. Before the mutex is released, the event will capture the search kernels. Hence later
      users can wait for the event again in their streams to copy the results back.
      */
      if (submitted_queries == free_spots || deadline <= std::chrono::system_clock::now()) {
        submit(*current_batch);
      }
      break;
    }

    // submit the rest of the queries if these didn't fit in the batch
    if (submitted_queries < queries.extent(0)) {
      RAFT_FAIL("This shouldn't happen during my testing (max_batch_size `mod` n_queries == 0)");
      auto rem_queries = queries.extent(0) - submitted_queries;
      search(
        res,
        params,
        raft::make_device_matrix_view(queries.data_handle() + submitted_queries * queries.extent(1),
                                      rem_queries,
                                      queries.extent(1)),
        raft::make_device_matrix_view(
          neighbors.data_handle() + submitted_queries * neighbors.extent(1),
          rem_queries,
          neighbors.extent(1)),
        raft::make_device_matrix_view(
          distances.data_handle() + submitted_queries * distances.extent(1),
          rem_queries,
          distances.extent(1)));
    }

    while (my_submit_counter == submit_counter.load()) {
      {
        std::lock_guard<std::mutex> guard(current_batch->lock);
        // Check the counter again, just in case
        if (my_submit_counter != submit_counter.load()) { break; }
        // Submit the query if the deadline has passed.
        if (deadline <= std::chrono::system_clock::now()) {
          submit(*current_batch);
          break;
        }
      }
      std::this_thread::sleep_until(
        deadline - std::chrono::nanoseconds(size_t(params.soft_deadline_ms * 500000.0)));
    }

    // TODO: it's not good that we wait for chunks in the reverse order
    //                                          (due to recursive search above)
    RAFT_CUDA_TRY(cudaStreamWaitEvent(user_stream, current_batch->event));
    raft::copy(neighbors.data_handle(),
               current_batch->neighbors.data_handle() + batch_offset * neighbors.extent(1),
               submitted_queries * neighbors.extent(1),
               user_stream);
    raft::copy(distances.data_handle(),
               current_batch->distances.data_handle() + batch_offset * distances.extent(1),
               submitted_queries * distances.extent(1),
               user_stream);
  }

 private:
  raft::resources res_;  // Sic! Store by value to copy the resource.
  std::function<function_search_type<T, IdxT>> upstream_search_;
  size_t n_queues_;

  mutable alignas(kCacheLineBytes) batch<T, IdxT>* batch_bufs_;
  mutable alignas(kCacheLineBytes) cuda::std::atomic<int64_t> submit_counter{0};

  void submit(batch<T, IdxT>& current_batch) const
  {
    auto batch_size = current_batch.size.load(cuda::memory_order_relaxed);
    auto queries    = raft::make_device_matrix_view<const T>(
      current_batch.queries.data_handle(), batch_size, current_batch.queries.extent(1));
    auto neighbors = raft::make_device_matrix_view<IdxT>(
      current_batch.neighbors.data_handle(), batch_size, current_batch.neighbors.extent(1));
    auto distances = raft::make_device_matrix_view<float>(
      current_batch.distances.data_handle(), batch_size, current_batch.distances.extent(1));
    current_batch.size.store(0, cuda::memory_order_relaxed);
    current_batch.deadline.store(batch<T, IdxT>::time_point::max(), cuda::memory_order_relaxed);

    upstream_search_(current_batch.res, queries, neighbors, distances);
    RAFT_CUDA_TRY(cudaEventRecord(current_batch.event, current_batch.stream));
    // ^ event captures the job by the time the submit counter is incremented
    submit_counter.fetch_add(1, cuda::memory_order_release);
  }
};

}  // namespace cuvs::neighbors::dynamic_batching::detail
