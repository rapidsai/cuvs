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
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/pinned_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>

#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda/std/atomic>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <chrono>
#include <limits>
#include <memory>
#include <vector>

#ifndef CUVS_SYSTEM_LITTLE_ENDIAN
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define CUVS_SYSTEM_LITTLE_ENDIAN 0
#else
#define CUVS_SYSTEM_LITTLE_ENDIAN 1
#endif
#endif

namespace cuvs::neighbors::dynamic_batching::detail {

class cuda_event {
 public:
  cuda_event(cuda_event&&)            = default;
  cuda_event& operator=(cuda_event&&) = default;
  ~cuda_event()                       = default;
  cuda_event(cuda_event const&)       = delete;  // Copying disallowed: one event one owner
  cuda_event& operator=(cuda_event&)  = delete;

  cuda_event()
    : event_{[]() {
               cudaEvent_t* e = new cudaEvent_t;
               RAFT_CUDA_TRY(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
               return e;
             }(),
             [](cudaEvent_t* e) {
               RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(*e));
               delete e;
             }}
  {
  }

  cudaEvent_t value() const { return *event_; }

 private:
  std::unique_ptr<cudaEvent_t, std::function<void(cudaEvent_t*)>> event_;
};

template <typename MdSpanOrArray>
struct get_accessor_type_t {
  using type = typename MdSpanOrArray::accessor_type;
};

template <typename ElementType, typename Extents, typename LayoutPolicy, typename ContainerPolicy>
struct get_accessor_type_t<raft::mdarray<ElementType, Extents, LayoutPolicy, ContainerPolicy>> {
  using mdarray_type = raft::mdarray<ElementType, Extents, LayoutPolicy, ContainerPolicy>;
  using view_type    = typename mdarray_type::view_type;
  using type         = typename view_type::accessor_type;
};

template <typename MdSpanOrArray>
using get_accessor_type = typename get_accessor_type_t<MdSpanOrArray>::type;

template <typename Source3DT>
constexpr inline auto slice_3d(typename Source3DT::index_type i, const Source3DT& source3d)
{
  using element_type  = typename Source3DT::element_type;
  using index_type    = typename Source3DT::index_type;
  using layout_type   = typename Source3DT::layout_type;
  using accessor_type = get_accessor_type<Source3DT>;
  auto extent2d       = raft::make_extents<index_type>(source3d.extent(1), source3d.extent(2));
  auto stride         = uint64_t(extent2d.extent(0)) * uint64_t(extent2d.extent(1));
  return raft::mdspan<element_type, decltype(extent2d), layout_type, accessor_type>{
    const_cast<element_type*>(source3d.data_handle()) + stride * i, extent2d};
}

template <typename Source2DT>
constexpr inline auto slice_2d(typename Source2DT::index_type i, const Source2DT& source2d)
{
  using element_type  = typename Source2DT::element_type;
  using index_type    = typename Source2DT::index_type;
  using layout_type   = typename Source2DT::layout_type;
  using accessor_type = get_accessor_type<Source2DT>;
  auto extent1d       = raft::make_extents<index_type>(source2d.extent(1));
  auto stride         = uint64_t(extent1d.extent(0));
  return raft::mdspan<element_type, decltype(extent1d), layout_type, accessor_type>{
    const_cast<element_type*>(source2d.data_handle()) + stride * i, extent1d};
}

// ---------------------------------------------

constexpr size_t kCacheLineBytes = 64;

template <typename Upstream, typename T, typename IdxT>
using upstream_search_type_const = void(raft::resources const&,
                                        typename Upstream::search_params_type const&,
                                        Upstream const&,
                                        raft::device_matrix_view<const T, int64_t, raft::row_major>,
                                        raft::device_matrix_view<IdxT, int64_t, raft::row_major>,
                                        raft::device_matrix_view<float, int64_t, raft::row_major>,
                                        const cuvs::neighbors::filtering::base_filter&);

template <typename Upstream, typename T, typename IdxT>
using upstream_search_type = void(raft::resources const&,
                                  typename Upstream::search_params_type const&,
                                  Upstream&,
                                  raft::device_matrix_view<const T, int64_t, raft::row_major>,
                                  raft::device_matrix_view<IdxT, int64_t, raft::row_major>,
                                  raft::device_matrix_view<float, int64_t, raft::row_major>,
                                  const cuvs::neighbors::filtering::base_filter&);

template <typename T, typename IdxT>
using function_search_type = void(raft::resources const&,
                                  raft::device_matrix_view<const T, int64_t, raft::row_major>,
                                  raft::device_matrix_view<IdxT, int64_t, raft::row_major>,
                                  raft::device_matrix_view<float, int64_t, raft::row_major>);

/**
 * Identifies the batch and its job-commit state.
 * Should be in the pinned memory for fast shared access on CPU and GPU side.
 */
struct batch_token {
  uint64_t value = 0;

  constexpr inline batch_token() {}
  explicit constexpr inline batch_token(uint32_t buffer_id) { id() = buffer_id; }

  /** Sequential id of the batch in the array of batches. */
  inline auto id() noexcept -> uint32_t&
  {
    return *(reinterpret_cast<uint32_t*>(&value) + kOffsetOfId);
  }
  /**
   * How many queries are promised by the participating CPU threads (requesters).
   * Any actor (CPU or GPU thread) may atomically add (max_batch_size+1) to this value (multiple
   * times even), which indicates that the actor cannot wait for more queries to come anymore.
   * Hence, the actual number of committed queries is `size_committed % (max_batch_size+1)`.
   *
   * The gather kernel cannot finish while `size_committed < max_batch_size`.
   */
  inline auto size_committed() noexcept -> uint32_t&
  {
    return *(reinterpret_cast<uint32_t*>(&value) + kOffsetOfSC);
  }

 private:
  /** Offset of the `id()` value in the token if it's interpreted as uint32_t[2]. */
  static constexpr inline uint32_t kOffsetOfId = CUVS_SYSTEM_LITTLE_ENDIAN;
  /** Offset of the `size_committed()` value in the token if it's interpreted as uint32_t[2]. */
  static constexpr inline uint32_t kOffsetOfSC = 1 - kOffsetOfId;
};
static_assert(sizeof(batch_token) == sizeof(uint64_t));
static_assert(cuda::std::atomic<batch_token>::is_always_lock_free);

template <uint32_t Size>
struct batch_queue_t {
  static constexpr uint32_t kSize        = Size;
  static constexpr uint32_t kMinElemSize = sizeof(uint32_t);
  static_assert(cuda::std::atomic<batch_token>::is_always_lock_free,
                "The value type must be lock-free.");
  static_assert(cuda::std::atomic<uint32_t>::is_always_lock_free,
                "The value type must be lock-free.");
  static_assert(cuda::std::atomic<int32_t>::is_always_lock_free,
                "The value type must be lock-free.");
  static_assert(raft::is_a_power_of_two(kSize), "The size must be a power-of-two for efficiency.");

  static constexpr auto kMemOrder = cuda::std::memory_order_relaxed;

  /** Type-safe synonym for the internal head & tail counters. */
  struct seq_order_id {
    uint32_t value;
  };

  explicit batch_queue_t(const raft::resources& res, uint32_t capacity = Size) noexcept
    : tokens_{raft::make_pinned_vector<cuda::atomic<batch_token, cuda::thread_scope_system>,
                                       uint32_t>(res, kSize)},
      rem_time_us_{
        raft::make_pinned_vector<cuda::atomic<int32_t, cuda::thread_scope_system>, uint32_t>(
          res, kSize)},
      dispatch_sequence_id_(kSize),
      capacity_{capacity}
  {
    tail_.store(0, kMemOrder);
    head_.store(0, kMemOrder);
    for (uint32_t i = 0; i < kSize; i++) {
      rem_time_us_(i).store(std::numeric_limits<int32_t>::max(), kMemOrder);
      dispatch_sequence_id_[i].store(uint32_t(-1), kMemOrder);
      tokens_(i).store(loop_invalid_token(seq_order_id{uint32_t(-1)}), kMemOrder);
    }
  }

  /** Nominal capacity of the queue. */
  [[nodiscard]] auto capacity() const { return capacity_; }

  /**
   * Advance the tail position, ensure the slot is empty, and return the reference to the new slot.
   * The calling side is responsible for filling-in the slot with an actual value at the later time.
   */
  inline auto push() noexcept -> cuda::atomic<batch_token, cuda::thread_scope_system>&
  {
    auto& loc = token(seq_order_id{tail_.fetch_add(1, kMemOrder)});
    while (loc.load(kMemOrder).value < 0xfffffffe00000000ull) {  // TODO TMP kEmpty.value
      // Wait till the slot becomes empty (doesn't matter future or past).
      // The batch id is only every updated in the scatter kernel, which is the only source of truth
      // whether a batch buffers are currently used by the GPU.
      std::this_thread::yield();
    }
    return loc;
  }

  /** Get the reference to the first element in the queue. */
  inline auto head() noexcept -> seq_order_id
  {
    auto h = head_.load(kMemOrder);
    // The head cannot go ahead of the tail by more than the queue buffer size.
    // If the head is ahead by not more than kSize elements though, everything is fine;
    // The slots too far ahead are protected by loop_invalid_token().
    while (static_cast<int32_t>(h - tail_.load(kMemOrder)) >= static_cast<int32_t>(kSize)) {
      std::this_thread::yield();
      h = head_.load(kMemOrder);
    }
    return seq_order_id{h};
  }

  /** Batch commit state and IO buffer id (see `batch_token`) */
  inline auto token(seq_order_id id) -> cuda::atomic<batch_token, cuda::thread_scope_system>&
  {
    return tokens_(cache_friendly_idx(id.value));
  }

  /**
   * How much time has this batch left for waiting.
   * It is an approximate value by design - to minimize the synchronization between CPU and GPU.
   *
   * The clocks on GPU and CPU may have different values, so the running kernel and the CPU thread
   * have different ideas on how much time is left. Rather than trying to synchronize the clocks, we
   * maintain independent timers and accept the uncertainty.
   *
   * Access pattern: CPU write-only (producer); GPU read-only (consumer).
   */
  inline auto rem_time_us(seq_order_id id) -> cuda::atomic<int32_t, cuda::thread_scope_system>&
  {
    return rem_time_us_(cache_friendly_idx(id.value));
  }

  /**
   * This value is updated by the host thread after it submits the job completion event to indicate
   * to other threads can wait on the event to get the results back.
   * Other threads get the value from the batch queue and compare that value against this atomic.
   *
   * Access pattern: CPU-only; dispatching thread writes the id once, other threads wait on it.
   */
  inline auto dispatch_sequence_id(seq_order_id id) -> cuda::std::atomic<uint32_t>&
  {
    return dispatch_sequence_id_[cache_friendly_idx(id.value)];
  }

  /**
   * An `atomicMax` on the queue head in disguise.
   * This makes the given batch slot and all prior slots unreachable (not possible to commit).
   */
  inline void pop(seq_order_id id) noexcept
  {
    const auto desired = id.value + 1;
    auto observed      = id.value;
    while (observed < desired && !head_.compare_exchange_weak(
                                   observed, std::max(observed, desired), kMemOrder, kMemOrder)) {}
  }

  /**
   * Whether this is an odd (or even) loop over the queue.
   * This information is used to distinguish between the empty slots available for committing
   * (future batches) and the invalid, already used slots ("empty past").
   */
  static constexpr inline auto loop_oddity(seq_order_id id) noexcept -> bool
  {
    return (id.value & kSize) > 0;
  }

  static constexpr inline auto loop_invalid_id(seq_order_id id) noexcept -> uint32_t
  {
    return loop_oddity(id) ? 0xffffffffu : 0xfffffffeu;
  }

  static constexpr inline auto loop_invalid_token(seq_order_id id) noexcept -> batch_token
  {
    return batch_token{loop_invalid_id(id)};
  }

 private:
  alignas(kCacheLineBytes) cuda::std::atomic<uint32_t> tail_{};
  alignas(kCacheLineBytes) cuda::std::atomic<uint32_t> head_{};

  alignas(kCacheLineBytes)
    raft::pinned_vector<cuda::atomic<batch_token, cuda::thread_scope_system>, uint32_t> tokens_;
  raft::pinned_vector<cuda::atomic<int32_t, cuda::thread_scope_system>, uint32_t> rem_time_us_;
  std::vector<cuda::std::atomic<uint32_t>> dispatch_sequence_id_;
  uint32_t capacity_;

  /* [Note: cache-friendly indexing]
     To avoid false sharing, the queue pushes and pops values not sequentially, but with an
     increment that is larger than the cache line size.
     Hence we introduce the `kCounterIncrement > kCacheLineBytes`.
     However, to make sure all indices are used, we choose the increment to be coprime with the
     buffer size. We also require that the buffer size is a power-of-two for two reasons:
       1) Fast modulus operation - reduces to binary `and` (with `kCounterLocMask`).
       2) Easy to ensure GCD(kCounterIncrement, kSize) == 1 by construction
          (see the definition below).
   */
  static constexpr uint32_t kElemsPerCacheLine =
    raft::div_rounding_up_safe<uint32_t>(kCacheLineBytes, kMinElemSize);
  static constexpr uint32_t kCounterIncrement = raft::bound_by_power_of_two(kElemsPerCacheLine) + 1;
  static constexpr uint32_t kCounterLocMask   = kSize - 1;
  // These props hold by design, but we add them here as a documentation and a sanity check.
  static_assert(
    kCounterIncrement * kMinElemSize >= kCacheLineBytes,
    "The counter increment should be larger than the cache line size to avoid false sharing.");
  static_assert(
    std::gcd(kCounterIncrement, kSize) == 1,
    "The counter increment and the size must be coprime to allow using all of the queue slots.");
  /** Map the sequential index onto cache-friendly strided index. */
  static constexpr inline auto cache_friendly_idx(uint32_t source_idx) noexcept -> uint32_t
  {
    return (source_idx * kCounterIncrement) & kCounterLocMask;
  }
};

template <typename T, typename IdxT>
struct alignas(kCacheLineBytes) request_pointers {
  /**
   * A pointer to `dim` values of a single query (input).
   *
   * Serves as a synchronization point between the CPU thread (producer) and a GPU block in the
   * `gather_inputs` kernel (consumer).
   */
  cuda::atomic<const T*, cuda::thread_scope_system> query{nullptr};
  /** A pointer to `k` nearest neighbors (output) */
  IdxT* neighbors{nullptr};
  /** A pointer to distances of `k` nearest neighbors (output) */
  float* distances{nullptr};
};

struct gpu_time_keeper {
  RAFT_DEVICE_INLINE_FUNCTION gpu_time_keeper(
    cuda::atomic<int32_t, cuda::thread_scope_system>* cpu_provided_remaining_time_us)
    : cpu_provided_remaining_time_us_{cpu_provided_remaining_time_us}
  {
    update_timestamp();
  }

  RAFT_DEVICE_INLINE_FUNCTION auto has_time() noexcept -> bool
  {
    if (timeout) { return false; }
    update_local_remaining_time();
    if (local_remaining_time_us_ <= 0) {
      timeout = true;
      return false;
    }
    update_cpu_provided_remaining_time();
    if (local_remaining_time_us_ <= 0) {
      timeout = true;
      return false;
    }
    return true;
  }

 private:
  cuda::atomic<int32_t, cuda::thread_scope_system>* cpu_provided_remaining_time_us_;
  uint64_t timestamp_ns_           = 0;
  int32_t local_remaining_time_us_ = std::numeric_limits<int32_t>::max();
  bool timeout                     = false;

  RAFT_DEVICE_INLINE_FUNCTION void update_timestamp() noexcept
  {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp_ns_));
  }

  RAFT_DEVICE_INLINE_FUNCTION void update_local_remaining_time() noexcept
  {
    auto prev_timestamp = timestamp_ns_;
    update_timestamp();
    // subtract the time passed since the last check
    // (assuming local time is updated every time timestamp is read)
    local_remaining_time_us_ -= static_cast<int32_t>((timestamp_ns_ - prev_timestamp) / 1000ull);
  }

  RAFT_DEVICE_INLINE_FUNCTION void update_cpu_provided_remaining_time() noexcept
  {
    local_remaining_time_us_ =
      std::min<int32_t>(local_remaining_time_us_,
                        cpu_provided_remaining_time_us_->load(cuda::std::memory_order_relaxed));
  }
};

/**
 * Copy the queries from the submitted pointers to the batch store, one query per block.
 * Upon completion of this kernel, the submitted queries are all in the contiguous buffer
 * `batch_queries`.
 *
 * Block size: (n, 1, 1) any number of threads copying a single row of data.
 * Grid size: (max_batch_size, 1, 1) - one block per query
 */
template <typename T, typename IdxT>
RAFT_KERNEL gather_inputs(
  raft::device_matrix_view<T, uint32_t, raft::row_major> batch_queries,
  raft::pinned_vector_view<request_pointers<T, IdxT>, uint32_t> request_ptrs,
  /* The remaining time may be updated on the host side: a thread with a tighter deadline may reduce
     it (but not increase). */
  cuda::atomic<int32_t, cuda::thread_scope_system>* remaining_time_us,
  /* This many queries are promised to be written into request_*_ptrs by host threads. */
  cuda::atomic<uint32_t, cuda::thread_scope_system>* batch_size_committed)
{
  const uint32_t query_id = blockIdx.x;
  __shared__ const T* query_ptr;
  volatile uint8_t* batch_fully_committed =
    reinterpret_cast<volatile uint8_t*>(batch_size_committed) + (CUVS_SYSTEM_LITTLE_ENDIAN * 3);
  volatile uint32_t* bs_committed = reinterpret_cast<volatile uint32_t*>(batch_size_committed);

  if (threadIdx.x == 0) {
    query_ptr = nullptr;
    gpu_time_keeper runtime{remaining_time_us};
    bool committed          = false;  // if the query is committed, we have to wait for it to arrive
    auto& request_query_ptr = request_ptrs(query_id).query;
    while (true) {
      query_ptr = request_query_ptr.load(cuda::std::memory_order_acquire);
      if (query_ptr != nullptr) {
        // The query is submitted to this block's slot; erase the pointer buffer for future use and
        // exit the loop.
        request_query_ptr.store(nullptr, cuda::std::memory_order_relaxed);
        break;
      }
      // The query hasn't been submitted, but is already committed; other checks may be skipped
      if (committed) { continue; }
      // Check if the query is committed
      // auto committed_count = batch_size_committed->load(cuda::std::memory_order_relaxed);
      uint32_t committed_count;
      asm volatile("ld.volatile.global.u32 %0, [%1];"
                   : "=r"(committed_count)
                   : "l"(bs_committed)
                   : "memory");
      committed = (committed_count & 0x00ffffff) > query_id;
      if (committed) { continue; }
      // If the query is not committed, but the batch is past the deadline, we exit without copying
      // the query
      if (committed_count > 0x00ffffff) { break; }
      // The query hasn't been submitted yet; check if we're past the deadline
      if (runtime.has_time()) { continue; }
      // Otherwise, let the others now time is out
      // Set the highest byte of the commit counter to 1 (thus avoiding RMW atomic)
      // This prevents any more CPU threads from committing to this batch.
      asm volatile("st.volatile.global.u8 [%0], %1;"
                   :
                   : "l"(batch_fully_committed), "r"(1)
                   : "memory");
      // committed_count = batch_size_committed->load(cuda::std::memory_order_seq_cst);
      asm volatile("ld.volatile.global.u32 %0, [%1];"
                   : "=r"(committed_count)
                   : "l"(bs_committed)
                   : "memory");
      committed = (committed_count & 0x00ffffff) > query_id;
      if (committed) { continue; }
      break;
    }
  }
  // The block waits till the leading thread gets the query pointer
  cooperative_groups::this_thread_block().sync();
  auto query_ptr_local = query_ptr;
  if (query_ptr_local == nullptr) { return; }
  // block-wide copy input query
  auto dim = batch_queries.extent(1);
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    batch_queries(query_id, i) = query_ptr_local[i];
  }
}

/** Copy the results of the search back to the requesters. */
template <typename T, typename IdxT>
RAFT_KERNEL scatter_outputs(
  raft::pinned_vector_view<request_pointers<T, IdxT>, uint32_t> request_ptrs,
  raft::device_matrix_view<const IdxT, uint32_t> batch_neighbors,
  raft::device_matrix_view<const float, uint32_t> batch_distances,
  cuda::atomic<int32_t, cuda::thread_scope_system>* remaining_time_us,
  cuda::atomic<uint64_t, cuda::thread_scope_system>* this_token,
  uint64_t empty_token_value,
  cuda::atomic<uint64_t, cuda::thread_scope_system>* next_token,
  uint32_t batch_id)
{
  __shared__ uint32_t batch_size;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    batch_size = this_token->load(cuda::std::memory_order_relaxed) & 0x00ffffff;
  }
  // Copy output
  cooperative_groups::this_thread_block().sync();
  auto k = batch_neighbors.extent(1);
  for (uint32_t i = threadIdx.y; i < batch_size; i += blockDim.y) {
    auto* request_neighbors = request_ptrs(i).neighbors;
    auto* request_distances = request_ptrs(i).distances;
    for (uint32_t j = threadIdx.x; j < k; j += blockDim.x) {
      request_neighbors[j] = batch_neighbors(i, j);
      request_distances[j] = batch_distances(i, j);
    }
  }
  // Clear the batch state after all threads copied the data, so the batch can be reused
  cooperative_groups::this_thread_block().sync();
  if (threadIdx.x != 0 || threadIdx.y != 0) { return; }
  remaining_time_us->store(std::numeric_limits<int32_t>::max(), cuda::std::memory_order_relaxed);
  reinterpret_cast<cuda::atomic<uint32_t, cuda::thread_scope_system>*>(
    next_token)[CUVS_SYSTEM_LITTLE_ENDIAN]
    .store(batch_id, cuda::std::memory_order_relaxed);
  this_token->store(empty_token_value, cuda::std::memory_order_release);
}

/**
 * Batch runner is shared among the users of the `dynamic_batching::index` (i.e. the index can be
 * copied, but the copies hold shared pointers to a single batch runner).
 *
 * Constructor and destructor of this class do not need to be thread-safe, as their execution is
 * guaranteed to happen in one thread by the holding shared pointer.
 *
 * The search function must be thread-safe. We only have to pay attention to the `mutable` members
 * though, because the function marked const.
 */
template <typename T, typename IdxT>
class batch_runner {
 public:
  constexpr static uint32_t kMaxNumQueues = 256;

  // Save the parameters and the upstream batched search function to invoke
  template <typename Upstream>
  batch_runner(const raft::resources& res,
               const dynamic_batching::index_params<Upstream>& params,
               upstream_search_type_const<Upstream, T, IdxT>* upstream_search)
    : res_{res},
      upstream_search_{[&ix = params.upstream,
                        &ps = params.upstream_params,
                        upstream_search,
                        sample_filter = params.sample_filter](
                         raft::resources const& res,
                         raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
                         raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,
                         raft::device_matrix_view<float, int64_t, raft::row_major> distances) {
        if (sample_filter == nullptr) {
          using base_filter_type = cuvs::neighbors::filtering::base_filter;
          const auto none_filter = cuvs::neighbors::filtering::none_sample_filter{};
          return upstream_search(res,
                                 ps,
                                 ix,
                                 queries,
                                 neighbors,
                                 distances,
                                 static_cast<const base_filter_type&>(none_filter));

        } else {
          return upstream_search(res, ps, ix, queries, neighbors, distances, *sample_filter);
        }
      }},
      k_{uint32_t(params.k)},
      dim_{uint32_t(params.dim)},
      max_batch_size_{uint32_t(params.max_batch_size)},
      n_queues_{uint32_t(params.n_queues)},
      batch_queue_{res_, n_queues_},
      completion_events_(n_queues_),
      input_extents_{n_queues_, max_batch_size_, dim_},
      output_extents_{n_queues_, max_batch_size_, k_},
      queries_{raft::make_device_mdarray<T>(res_, input_extents_)},
      neighbors_{raft::make_device_mdarray<IdxT>(res_, output_extents_)},
      distances_{raft::make_device_mdarray<float>(res_, output_extents_)},
      request_ptrs_{raft::make_pinned_matrix<request_pointers<T, IdxT>, uint32_t>(
        res_, n_queues_, max_batch_size_)}
  {
    // Make sure to initialize the atomic values in the batch_state structs.
    for (uint32_t i = 0; i < n_queues_; i++) {
      batch_queue_.push().store(batch_token{i});
      // Make sure to initialize query pointers, because they are used for synchronization
      for (uint32_t j = 0; j < max_batch_size_; j++) {
        new (&request_ptrs_(i, j)) request_pointers<T, IdxT>{};
      }
    }
  }

  // A workaround for algos, which have non-const `index` type in their arguments
  template <typename Upstream>
  batch_runner(const raft::resources& res,
               const dynamic_batching::index_params<Upstream>& params,
               upstream_search_type<Upstream, T, IdxT>* upstream_search)
    : batch_runner{
        res,
        params,
        reinterpret_cast<upstream_search_type_const<Upstream, T, IdxT>*>(upstream_search)}
  {
  }

  void search(raft::resources const& res,
              cuvs::neighbors::dynamic_batching::search_params const& params,
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
              raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,
              raft::device_matrix_view<float, int64_t, raft::row_major> distances) const
  {
    uint32_t n_queries = queries.extent(0);
    if (n_queries >= max_batch_size_) {
      return upstream_search_(res, queries, neighbors, distances);
    }

    auto deadline = std::chrono::system_clock::now() +
                    std::chrono::nanoseconds(size_t(params.soft_deadline_ms * 1000000.0));

    int64_t local_io_offset = 0;
    batch_token batch_token_observed{0};
    while (true) {
      const auto seq_id            = batch_queue_.head();
      const auto commit_result     = try_commit(seq_id, n_queries);
      const auto queries_committed = std::get<uint32_t>(commit_result);
      if (queries_committed == 0) {
        // try to get a new batch
        continue;
      }
      batch_token_observed           = std::get<batch_token>(commit_result);
      const auto batch_offset        = batch_token_observed.size_committed();
      auto& batch_token_ref          = batch_queue_.token(seq_id);
      auto& rem_time_us_ref          = batch_queue_.rem_time_us(seq_id);
      auto& dispatch_sequence_id_ref = batch_queue_.dispatch_sequence_id(seq_id);
      for (uint64_t wait_iteration = 0; batch_token_observed.id() >= n_queues_; wait_iteration++) {
        /* Note: waiting for batch IO buffers
        The CPU threads can commit to the incoming batches in the queue in advance (this happens in
        try_commit).
        In this loop, a thread waits for the batch IO buffer to be released by a running search on
        the GPU side (scatter_outputs kernel). Hence, this loop is engaged only if all buffers are
        currently used, which suggests that the GPU is busy (or there's not enough IO buffers).
        This also means the current search is not likely to meet the deadline set by the user.

        The scatter kernel returns its buffer id into an acquired slot in the batch queue; in this
        loop we wait for that id to arrive.

        Generally, we want to waste as little as possible CPU cycles here to let other threads wait
        on dispatch_sequence_id_ref below more efficiently. At the same time, we shouldn't use
        `.wait()` here, because `.notify_all()` would have to come from GPU.
        */
        if (wait_iteration < 2) {
          // Don't wait and try get the value asap
        } else if (wait_iteration < 20) {
          std::this_thread::yield();
        } else {
          // sleep for 1/10 of deadline time or more
          std::this_thread::sleep_for(
            std::chrono::nanoseconds(size_t(params.soft_deadline_ms * 100000.0) *
                                     raft::div_rounding_up_safe<uint64_t>(wait_iteration, 100)));
        }
        batch_token_observed = batch_token_ref.load(cuda::std::memory_order_acquire);
      }
      // Whether this thread is responsible for dispatching the batch.
      bool is_dispatcher = batch_offset == 0;
      auto stream        = raft::resource::get_cuda_stream(res);
      auto batch_id      = batch_token_observed.id();
      auto request_ptrs  = slice_2d(batch_id, request_ptrs_);

      if (is_dispatcher) {
        // run the gather kernel before submitting the data to reduce the latency
        gather_inputs<T, IdxT><<<max_batch_size_, 32, 0, stream>>>(
          slice_3d(batch_id, queries_),
          request_ptrs,
          &rem_time_us_ref,
          reinterpret_cast<cuda::atomic<uint32_t, cuda::thread_scope_system>*>(
            &reinterpret_cast<batch_token*>(&batch_token_ref)->size_committed()));
      }

      // *** Set the pointers to queries, neighbors, distances - query-by-query
      for (uint32_t i = 0; i < queries_committed; i++) {
        const auto o   = local_io_offset + i;
        auto& ptrs     = request_ptrs(batch_offset + i);
        ptrs.neighbors = neighbors.data_handle() + o * k_;
        ptrs.distances = distances.data_handle() + o * k_;
        ptrs.query.store(queries.data_handle() + o * dim_, cuda::std::memory_order_release);
      }

      // Submit estimated remaining time
      {
        auto rem_time_us = static_cast<int32_t>(
          std::max<int64_t>(0, (deadline - std::chrono::system_clock::now()).count()) / 1000);
        rem_time_us_ref.fetch_min(rem_time_us, cuda::std::memory_order_relaxed);
      }

      if (is_dispatcher) {
        auto batch_neighbors = slice_3d(batch_id, neighbors_);
        auto batch_distances = slice_3d(batch_id, distances_);
        upstream_search_(res, slice_3d(batch_id, queries_), batch_neighbors, batch_distances);
        auto& next_token_ref = batch_queue_.push();
        // next_batch_token);
        auto bs = dim3(128, 8, 1);
        scatter_outputs<T, IdxT><<<1, bs, 0, stream>>>(
          request_ptrs,
          batch_neighbors,
          batch_distances,
          &rem_time_us_ref,
          reinterpret_cast<cuda::atomic<uint64_t, cuda::thread_scope_system>*>(&batch_token_ref),
          batch_queue_.loop_invalid_token(seq_id).value,
          reinterpret_cast<cuda::atomic<uint64_t, cuda::thread_scope_system>*>(&next_token_ref),
          batch_id);
        RAFT_CUDA_TRY(cudaEventRecord(completion_events_[batch_id].value(), stream));
        dispatch_sequence_id_ref.store(seq_id.value, cuda::std::memory_order_release);
        dispatch_sequence_id_ref.notify_all();

      } else {
        // Wait till the dispatch_sequence_id counter is updated, which means the event is recorded
        auto dispatched_id_observed =
          dispatch_sequence_id_ref.load(cuda::std::memory_order_acquire);
        while (static_cast<int32_t>(seq_id.value - dispatched_id_observed) > 0) {
          dispatch_sequence_id_ref.wait(dispatched_id_observed, cuda::std::memory_order_relaxed);
          dispatched_id_observed = dispatch_sequence_id_ref.load(cuda::std::memory_order_acquire);
        }
        // Now we can safely record the event
        RAFT_CUDA_TRY(cudaStreamWaitEvent(stream, completion_events_[batch_id].value()));
      }

      n_queries -= queries_committed;

      if (n_queries == 0) { return; }
      // If not all queries were committed, continue in the loop.
      // TODO: it could potentially be more efficient to first commit everything and only then
      //        submit the work/wait for the event
      local_io_offset += queries_committed;
    }
  }

 private:
  raft::resources res_;  // Sic! Store by value to copy the resource.
  std::function<function_search_type<T, IdxT>> upstream_search_;
  uint32_t k_;
  uint32_t dim_;
  uint32_t max_batch_size_;
  uint32_t n_queues_;

  mutable batch_queue_t<kMaxNumQueues> batch_queue_;
  std::vector<cuda_event> completion_events_;

  using batch_extents = raft::extent_3d<uint32_t>;
  batch_extents input_extents_;
  batch_extents output_extents_;

  mutable raft::device_mdarray<T, batch_extents, raft::row_major> queries_;
  mutable raft::device_mdarray<IdxT, batch_extents, raft::row_major> neighbors_;
  mutable raft::device_mdarray<float, batch_extents, raft::row_major> distances_;

  mutable raft::pinned_matrix<request_pointers<T, IdxT>, uint32_t, raft::row_major> request_ptrs_;

  /**
   * Try to commit n_queries at most; returns the last observed batch_token (where `size_committed`
   * represents offset at which new queries are committed if successful) and the number of committed
   * queries.
   */
  auto try_commit(batch_queue_t<kMaxNumQueues>::seq_order_id seq_id,
                  uint32_t n_queries) const noexcept -> std::tuple<batch_token, uint32_t>
  {
    auto& batch_token_ref            = batch_queue_.token(seq_id);
    batch_token batch_token_observed = batch_token_ref.load();
    batch_token batch_token_updated  = batch_token_observed;
    // We have two values indicating the empty slot and switch between them once a full loop over
    // the token buffer. This allows us to distinguish between the empty future slots (into which a
    // thread can commit) and empty past slots (which are invalid).
    uint32_t empty_past = batch_queue_.loop_invalid_id(seq_id);
    do {
      // If the slot was recently used and now empty, it is an indication that the queue head
      // counter is outdated due to batches being finalized by the kernel (by the timeout).
      // That means we need to update the head counter and find a new slot to commit.
      if (batch_token_observed.size_committed() >= max_batch_size_ ||
          batch_token_observed.id() == empty_past) {
        batch_queue_.pop(seq_id);
        return std::make_tuple(batch_token_observed, 0);
      }
      batch_token_updated.id() = batch_token_observed.id();
      batch_token_updated.size_committed() =
        std::min(batch_token_observed.size_committed() + n_queries, max_batch_size_);
    } while (!batch_token_ref.compare_exchange_weak(batch_token_observed,
                                                    batch_token_updated,
                                                    cuda::std::memory_order_acq_rel,
                                                    cuda::std::memory_order_relaxed));
    if (batch_token_updated.size_committed() >= max_batch_size_) {
      // The batch is already full, let's try to pop it from the queue
      //                                 (if nobody has done so already)
      batch_queue_.pop(seq_id);
    }
    return std::make_tuple(
      batch_token_observed,
      batch_token_updated.size_committed() - batch_token_observed.size_committed());
  }
};

}  // namespace cuvs::neighbors::dynamic_batching::detail
