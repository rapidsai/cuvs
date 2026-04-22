/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../smem_utils.cuh"

#include <iostream>
#include <typeinfo>

// Include tags header before any other includes that might open namespaces
#include <cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>

#include "compute_distance.hpp"  // For dataset_descriptor_host
#include "jit_lto_kernels/cagra_jit_launcher_factory.hpp"
#include "jit_lto_kernels/hashmap.hpp"
#include "jit_lto_kernels/kernel_def.hpp"
#include "sample_filter_utils.cuh"  // For CagraSampleFilterWithQueryIdOffset
#include "search_plan.cuh"          // For search_params
#include "search_single_cta_kernel_launcher_common.cuh"
#include "shared_launcher_jit.hpp"  // For shared JIT helper functions

#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuvs/detail/jit_lto/AlgorithmLauncher.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace cuvs::neighbors::cagra::detail::single_cta_search {

// Persistent queues / runner (host). worker_handle_t, job_desc_t, kCacheLineBytes, k* job limits:
// `jit_lto_kernels/search_single_cta_device_helpers.cuh` via `kernel_def.hpp`.

template <typename T, uint32_t Size, T Empty = std::numeric_limits<T>::max()>
struct alignas(kCacheLineBytes) resource_queue_t {
  using value_type                   = T;
  static constexpr uint32_t kSize    = Size;
  static constexpr value_type kEmpty = Empty;
  static_assert(cuda::std::atomic<value_type>::is_always_lock_free,
                "The value type must be lock-free.");
  static_assert(raft::is_a_power_of_two(kSize), "The size must be a power-of-two for efficiency.");
  static constexpr uint32_t kElemsPerCacheLine =
    raft::div_rounding_up_safe<uint32_t>(kCacheLineBytes, sizeof(value_type));
  static constexpr uint32_t kCounterIncrement = raft::bound_by_power_of_two(kElemsPerCacheLine) + 1;
  static constexpr uint32_t kCounterLocMask   = kSize - 1;
  static_assert(
    kCounterIncrement * sizeof(value_type) >= kCacheLineBytes,
    "The counter increment should be larger than the cache line size to avoid false sharing.");
  static_assert(
    std::gcd(kCounterIncrement, kSize) == 1,
    "The counter increment and the size must be coprime to allow using all of the queue slots.");

  static constexpr auto kMemOrder = cuda::std::memory_order_relaxed;

  explicit resource_queue_t(uint32_t capacity = Size) noexcept : capacity_{capacity}
  {
    head_.store(0, kMemOrder);
    tail_.store(0, kMemOrder);
    for (uint32_t i = 0; i < kSize; i++) {
      buf_[i].store(kEmpty, kMemOrder);
    }
  }

  [[nodiscard]] auto capacity() const { return capacity_; }

  void set_capacity(uint32_t capacity) { capacity_ = capacity; }

  struct promise_t {
    explicit promise_t(cuda::std::atomic<value_type>& loc) : loc_{loc}, val_{Empty} {}
    ~promise_t() noexcept { wait(); }

    auto test() noexcept -> bool
    {
      if (val_ != Empty) { return true; }
      val_ = loc_.exchange(kEmpty, kMemOrder);
      return val_ != Empty;
    }

    auto test(value_type& e) noexcept -> bool
    {
      if (test()) {
        e = val_;
        return true;
      }
      return false;
    }

    auto wait() noexcept -> value_type
    {
      if (val_ == Empty) {
        do {
          loc_.wait(kEmpty, kMemOrder);
          val_ = loc_.exchange(kEmpty, kMemOrder);
        } while (val_ == Empty);
      }
      return val_;
    }

   private:
    cuda::std::atomic<value_type>& loc_;
    value_type val_;
  };

  void push(value_type x) noexcept
  {
    auto& loc    = buf_[head_.fetch_add(kCounterIncrement, kMemOrder) & kCounterLocMask];
    value_type e = kEmpty;
    while (!loc.compare_exchange_weak(e, x, kMemOrder, kMemOrder)) {
      e = kEmpty;
    }
    loc.notify_one();
  }

  auto pop() noexcept -> promise_t
  {
    auto& loc = buf_[tail_.fetch_add(kCounterIncrement, kMemOrder) & kCounterLocMask];
    return promise_t{loc};
  }

 private:
  alignas(kCacheLineBytes) cuda::std::atomic<uint32_t> head_{};
  alignas(kCacheLineBytes) cuda::std::atomic<uint32_t> tail_{};
  alignas(kCacheLineBytes) std::array<cuda::std::atomic<value_type>, kSize> buf_{};
  alignas(kCacheLineBytes) uint32_t capacity_;
};

template <typename T>
struct local_deque_t {
  explicit local_deque_t(uint32_t size) : store_(size) {}

  [[nodiscard]] auto capacity() const -> uint32_t { return store_.size(); }
  [[nodiscard]] auto size() const -> uint32_t { return end_ - start_; }

  void push_back(T x) { store_[end_++ % capacity()] = x; }

  void push_front(T x)
  {
    if (start_ == 0) {
      start_ += capacity();
      end_ += capacity();
    }
    store_[--start_ % capacity()] = x;
  }

  auto pop_back() -> T { return store_[--end_ % capacity()]; }
  auto pop_front() -> T { return store_[start_++ % capacity()]; }

  auto try_push_back(T x) -> bool
  {
    if (size() >= capacity()) { return false; }
    push_back(x);
    return true;
  }

  auto try_push_front(T x) -> bool
  {
    if (size() >= capacity()) { return false; }
    push_front(x);
    return true;
  }

  auto try_pop_back(T& x) -> bool
  {
    if (start_ >= end_) { return false; }
    x = pop_back();
    return true;
  }

  auto try_pop_front(T& x) -> bool
  {
    if (start_ >= end_) { return false; }
    x = pop_front();
    return true;
  }

 private:
  std::vector<T> store_;
  uint32_t start_{0};
  uint32_t end_{0};
};

struct persistent_runner_base_t {
  using job_queue_type    = resource_queue_t<uint32_t, kMaxJobsNum>;
  using worker_queue_type = resource_queue_t<uint32_t, kMaxWorkersNum>;
  rmm::mr::pinned_host_memory_resource worker_handles_mr;
  rmm::mr::pinned_host_memory_resource job_descriptor_mr;
  rmm::mr::cuda_memory_resource device_mr;
  cudaStream_t stream{};
  job_queue_type job_queue{};
  worker_queue_type worker_queue{};
  std::chrono::milliseconds lifetime;

  persistent_runner_base_t(float persistent_lifetime)
    : lifetime(size_t(persistent_lifetime * 1000)), job_queue(), worker_queue()
  {
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }
  virtual ~persistent_runner_base_t() noexcept { cudaStreamDestroy(stream); };
};

struct alignas(kCacheLineBytes) persistent_state {
  std::shared_ptr<persistent_runner_base_t> runner{nullptr};
  std::mutex lock;
};

inline persistent_state persistent{};

// Forward declarations
template <typename RunnerT, typename... Args>
auto get_runner_jit(Args... args) -> std::shared_ptr<RunnerT>;

template <typename RunnerT, typename... Args>
auto create_runner_jit(Args... args) -> std::shared_ptr<RunnerT>;

// Helper functions are now in shared_launcher_jit.hpp

// JIT-compatible launcher_t that works with worker_handle_t (same as non-JIT version)
struct alignas(kCacheLineBytes) launcher_jit_t {
  using job_queue_type           = resource_queue_t<uint32_t, kMaxJobsNum>;
  using worker_queue_type        = resource_queue_t<uint32_t, kMaxWorkersNum>;
  using pending_reads_queue_type = local_deque_t<uint32_t>;
  using completion_flag_type     = cuda::atomic<bool, cuda::thread_scope_system>;

  pending_reads_queue_type pending_reads;
  job_queue_type& job_ids;
  worker_queue_type& idle_worker_ids;
  worker_handle_t* worker_handles;
  uint32_t job_id;
  completion_flag_type* completion_flag;
  bool all_done = false;

  static inline constexpr auto kDefaultLatency = std::chrono::nanoseconds(50000);
  static inline constexpr auto kMaxExpectedLatency =
    kDefaultLatency * std::max<std::uint32_t>(10, kMaxJobsNum / 128);
  static inline thread_local auto expected_latency = kDefaultLatency;
  const std::chrono::time_point<std::chrono::system_clock> start;
  std::chrono::time_point<std::chrono::system_clock> now;
  const int64_t pause_factor;
  int pause_count = 0;
  std::chrono::time_point<std::chrono::system_clock> deadline;

  template <typename RecordWork>
  launcher_jit_t(job_queue_type& job_ids,
                 worker_queue_type& idle_worker_ids,
                 worker_handle_t* worker_handles,
                 uint32_t n_queries,
                 std::chrono::milliseconds max_wait_time,
                 RecordWork record_work)
    : pending_reads{std::min(n_queries, kMaxWorkersPerThread)},
      job_ids{job_ids},
      idle_worker_ids{idle_worker_ids},
      worker_handles{worker_handles},
      job_id{job_ids.pop().wait()},
      completion_flag{record_work(job_id)},
      start{std::chrono::system_clock::now()},
      pause_factor{calc_pause_factor(n_queries)},
      now{start},
      deadline{start + max_wait_time + expected_latency}
  {
    submit_query(idle_worker_ids.pop().wait(), 0);
    for (uint32_t i = 1; i < n_queries; i++) {
      auto promised_worker = idle_worker_ids.pop();
      uint32_t worker_id;
      while (!promised_worker.test(worker_id)) {
        if (pending_reads.try_pop_front(worker_id)) {
          bool returned_some = false;
          for (bool keep_returning = true; keep_returning;) {
            if (try_return_worker(worker_id)) {
              keep_returning = pending_reads.try_pop_front(worker_id);
              returned_some  = true;
            } else {
              pending_reads.push_front(worker_id);
              keep_returning = false;
            }
          }
          if (!returned_some) { pause(); }
        } else {
          worker_id = promised_worker.wait();
          break;
        }
      }
      pause_count = 0;
      submit_query(worker_id, i);
      if (i >= kSoftMaxWorkersPerThread && pending_reads.try_pop_front(worker_id)) {
        if (!try_return_worker(worker_id)) { pending_reads.push_front(worker_id); }
      }
    }
  }

  inline ~launcher_jit_t() noexcept
  {
    constexpr size_t kWindow = 100;
    expected_latency         = std::min<std::chrono::nanoseconds>(
      ((kWindow - 1) * expected_latency + now - start) / kWindow, kMaxExpectedLatency);
    if (job_id != job_queue_type::kEmpty) { job_ids.push(job_id); }
    uint32_t worker_id;
    while (pending_reads.try_pop_front(worker_id)) {
      idle_worker_ids.push(worker_id);
    }
  }

  inline void submit_query(uint32_t worker_id, uint32_t query_id)
  {
    worker_handles[worker_id].data.store(worker_handle_t::data_t{.value = {job_id, query_id}},
                                         cuda::memory_order_relaxed);
    while (!pending_reads.try_push_back(worker_id)) {
      auto pending_worker_id = pending_reads.pop_front();
      while (!try_return_worker(pending_worker_id)) {
        pause();
      }
    }
    pause_count = 0;
  }

  inline auto try_return_worker(uint32_t worker_id) -> bool
  {
    if (all_done ||
        !is_worker_busy(worker_handles[worker_id].data.load(cuda::memory_order_relaxed).handle)) {
      idle_worker_ids.push(worker_id);
      return true;
    } else {
      return false;
    }
  }

  inline auto is_all_done()
  {
    if (all_done) { return true; }
    all_done = completion_flag->load(cuda::memory_order_relaxed);
    return all_done;
  }

  [[nodiscard]] inline auto sleep_limit() const
  {
    constexpr auto kMinWakeTime  = std::chrono::nanoseconds(10000);
    constexpr double kSleepLimit = 0.6;
    return start + expected_latency * kSleepLimit - kMinWakeTime;
  }

  [[nodiscard]] inline auto overtime_threshold() const
  {
    constexpr auto kOvertimeFactor = 3;
    return start + expected_latency * kOvertimeFactor;
  }

  [[nodiscard]] inline auto calc_pause_factor(uint32_t n_queries) const -> uint32_t
  {
    constexpr uint32_t kMultiplier = 10;
    return kMultiplier * raft::div_rounding_up_safe(n_queries, idle_worker_ids.capacity());
  }

  inline void pause()
  {
    constexpr auto kSpinLimit    = 3;
    constexpr auto kPauseTimeMin = std::chrono::nanoseconds(1000);
    constexpr auto kPauseTimeMax = std::chrono::nanoseconds(50000);
    if (pause_count++ < kSpinLimit) {
      std::this_thread::yield();
      return;
    }
    now                  = std::chrono::system_clock::now();
    auto pause_time_base = std::max(now - start, expected_latency);
    auto pause_time      = std::clamp(pause_time_base / pause_factor, kPauseTimeMin, kPauseTimeMax);
    if (now + pause_time < sleep_limit()) {
      std::this_thread::sleep_for(pause_time);
    } else if (now <= overtime_threshold()) {
      std::this_thread::yield();
    } else if (now <= deadline) {
      std::this_thread::sleep_for(pause_time);
    } else {
      throw raft::exception(
        "The calling thread didn't receive the results from the persistent CAGRA kernel within the "
        "expected kernel lifetime. Here are possible reasons of this failure:\n"
        "  (1) `persistent_lifetime` search parameter is too small - increase it;\n"
        "  (2) there is other work being executed on the same device and the kernel failed to "
        "progress - decreasing `persistent_device_usage` may help (but not guaranteed);\n"
        "  (3) there is a bug in the implementation - please report it to cuVS team.");
    }
  }

  inline void wait()
  {
    uint32_t worker_id;
    while (pending_reads.try_pop_front(worker_id)) {
      while (!try_return_worker(worker_id)) {
        if (!is_all_done()) { pause(); }
      }
    }
    pause_count = 0;
    now         = std::chrono::system_clock::now();
    while (!is_all_done()) {
      auto till_time = sleep_limit();
      if (now < till_time) {
        std::this_thread::sleep_until(till_time);
        now = std::chrono::system_clock::now();
      } else {
        pause();
      }
    }
    job_ids.push(job_id);
    job_id = job_queue_type::kEmpty;
  }
};

// JIT persistent runner - uses AlgorithmLauncher instead of kernel function pointer
template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterT>
struct alignas(kCacheLineBytes) persistent_runner_jit_t : public persistent_runner_base_t {
  using index_type    = IndexT;
  using distance_type = DistanceT;
  using data_type     = DataT;
  // Must match job_desc_t<job_desc_traits<...>> in kernel_def.hpp / persistent kernel.
  using job_desc_type = job_desc_t<job_desc_traits<DataT, IndexT, DistanceT>>;

  std::shared_ptr<AlgorithmLauncher> launcher;
  uint32_t block_size;
  dataset_descriptor_host<DataT, IndexT, DistanceT> dd_host;
  rmm::device_uvector<worker_handle_t> worker_handles;
  rmm::device_uvector<job_desc_type> job_descriptors;
  rmm::device_uvector<uint32_t> completion_counters;
  rmm::device_uvector<index_type> hashmap;
  std::atomic<std::chrono::time_point<std::chrono::system_clock>> last_touch;
  uint64_t param_hash;
  cagra_bitset<SourceIndexT> bitset;

  static inline auto calculate_parameter_hash(
    std::reference_wrapper<const dataset_descriptor_host<DataT, IndexT, DistanceT>> dataset_desc,
    raft::device_matrix_view<const index_type, int64_t, raft::row_major> graph,
    const SourceIndexT* source_indices_ptr,
    uint32_t max_candidates,
    uint32_t num_itopk_candidates,
    uint32_t block_size,
    uint32_t smem_size,
    int64_t hash_bitlen,
    size_t small_hash_bitlen,
    size_t small_hash_reset_interval,
    uint32_t num_random_samplings,
    uint64_t rand_xor_mask,
    uint32_t num_seeds,
    uint32_t max_itopk,
    size_t itopk_size,
    size_t search_width,
    size_t min_iterations,
    size_t max_iterations,
    SampleFilterT sample_filter,
    float persistent_lifetime,
    float persistent_device_usage,
    std::shared_ptr<AlgorithmLauncher> /* launcher_ptr - not part of hash */,
    const void* /* dataset_desc - not part of hash */) -> uint64_t
  {
    return uint64_t(graph.data_handle()) ^ uint64_t(source_indices_ptr) ^
           dataset_desc.get().team_size ^ num_itopk_candidates ^ block_size ^ smem_size ^
           hash_bitlen ^ small_hash_reset_interval ^ num_random_samplings ^ rand_xor_mask ^
           num_seeds ^ itopk_size ^ search_width ^ min_iterations ^ max_iterations ^
           uint64_t(persistent_lifetime * 1000) ^ uint64_t(persistent_device_usage * 1000);
  }

  persistent_runner_jit_t(
    std::reference_wrapper<const dataset_descriptor_host<DataT, IndexT, DistanceT>> dataset_desc,
    raft::device_matrix_view<const index_type, int64_t, raft::row_major> graph,
    const SourceIndexT* source_indices_ptr,
    uint32_t max_candidates,
    uint32_t num_itopk_candidates,
    uint32_t block_size,
    uint32_t smem_size,
    int64_t hash_bitlen,
    size_t small_hash_bitlen,
    size_t small_hash_reset_interval,
    uint32_t num_random_samplings,
    uint64_t rand_xor_mask,
    uint32_t num_seeds,
    uint32_t max_itopk,
    size_t itopk_size,
    size_t search_width,
    size_t min_iterations,
    size_t max_iterations,
    SampleFilterT sample_filter,
    float persistent_lifetime,
    float persistent_device_usage,
    std::shared_ptr<AlgorithmLauncher> launcher_ptr,
    const void* /* dataset_desc - descriptor contains all needed info */)
    : persistent_runner_base_t{persistent_lifetime},
      launcher{launcher_ptr},
      block_size{block_size},
      worker_handles(0, stream, worker_handles_mr),
      job_descriptors(kMaxJobsNum, stream, job_descriptor_mr),
      completion_counters(kMaxJobsNum, stream, device_mr),
      hashmap(0, stream, device_mr),
      dd_host{dataset_desc.get()},
      param_hash(calculate_parameter_hash(dd_host,
                                          graph,
                                          source_indices_ptr,
                                          max_candidates,
                                          num_itopk_candidates,
                                          block_size,
                                          smem_size,
                                          hash_bitlen,
                                          small_hash_bitlen,
                                          small_hash_reset_interval,
                                          num_random_samplings,
                                          rand_xor_mask,
                                          num_seeds,
                                          max_itopk,
                                          itopk_size,
                                          search_width,
                                          min_iterations,
                                          max_iterations,
                                          sample_filter,
                                          persistent_lifetime,
                                          persistent_device_usage,
                                          launcher_ptr,
                                          nullptr))  // descriptor not needed in hash
  {
    const auto bf                  = extract_cagra_sample_filter<SourceIndexT>(sample_filter);
    this->bitset                   = bf.bitset;
    const uint32_t query_id_offset = bf.query_id_offset;

    // set kernel launch parameters
    dim3 gs = calc_coop_grid_size(block_size, smem_size, persistent_device_usage);
    dim3 bs(block_size, 1, 1);
    RAFT_LOG_DEBUG(
      "Launching JIT persistent kernel with %u threads, %u block %u smem", bs.x, gs.y, smem_size);

    // initialize the job queue
    auto* completion_counters_ptr = completion_counters.data();
    auto* job_descriptors_ptr     = job_descriptors.data();
    for (uint32_t i = 0; i < kMaxJobsNum; i++) {
      auto& jd                = job_descriptors_ptr[i].input.value;
      jd.result_indices_ptr   = 0;
      jd.result_distances_ptr = nullptr;
      jd.queries_ptr          = nullptr;
      jd.top_k                = 0;
      jd.n_queries            = 0;
      job_descriptors_ptr[i].completion_flag.store(false);
      job_queue.push(i);
    }

    // initialize the worker queue
    worker_queue.set_capacity(gs.y);
    worker_handles.resize(gs.y, stream);
    auto* worker_handles_ptr = worker_handles.data();
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    for (uint32_t i = 0; i < gs.y; i++) {
      worker_handles_ptr[i].data.store({kWaitForWork});
      worker_queue.push(i);
    }

    index_type* hashmap_ptr = nullptr;
    if (small_hash_bitlen == 0) {
      hashmap.resize(gs.y * hashmap::get_size(hash_bitlen), stream);
      hashmap_ptr = hashmap.data();
    }

    // Prepare kernel arguments
    // Get the device descriptor pointer - kernel will use the concrete type from template
    const auto* dev_desc = dataset_desc.get().dev_ptr(stream);

    // Cast size_t/int64_t parameters to match kernel signature exactly
    // The dispatch mechanism uses void* pointers, so parameter sizes must match exactly
    const uint32_t graph_degree_u32              = static_cast<uint32_t>(graph.extent(1));
    const uint32_t hash_bitlen_u32               = static_cast<uint32_t>(hash_bitlen);
    const uint32_t small_hash_bitlen_u32         = static_cast<uint32_t>(small_hash_bitlen);
    const uint32_t small_hash_reset_interval_u32 = static_cast<uint32_t>(small_hash_reset_interval);
    const uint32_t itopk_size_u32                = static_cast<uint32_t>(itopk_size);
    const uint32_t search_width_u32              = static_cast<uint32_t>(search_width);
    const uint32_t min_iterations_u32            = static_cast<uint32_t>(min_iterations);
    const uint32_t max_iterations_u32            = static_cast<uint32_t>(max_iterations);
    const unsigned num_random_samplings_u        = static_cast<unsigned>(num_random_samplings);

    const IndexT* seed_ptr_arg            = nullptr;
    uint32_t* num_executed_iterations_arg = nullptr;
    // Launch the persistent kernel via AlgorithmLauncher
    // The persistent kernel now takes the descriptor pointer directly
    launcher->dispatch_cooperative<
      single_cta_search::search_single_cta_p_kernel_func_t<DataT, IndexT, DistanceT, SourceIndexT>>(
      stream,
      gs,
      bs,
      static_cast<std::size_t>(smem_size),
      worker_handles_ptr,
      job_descriptors_ptr,
      completion_counters_ptr,
      graph.data_handle(),
      graph_degree_u32,  // Cast int64_t to uint32_t
      source_indices_ptr,
      num_random_samplings_u,  // Cast uint32_t to unsigned for consistency
      rand_xor_mask,           // uint64_t matches kernel (8 bytes)
      seed_ptr_arg,
      num_seeds,
      hashmap_ptr,
      max_candidates,
      max_itopk,
      itopk_size_u32,      // Cast size_t to uint32_t
      search_width_u32,    // Cast size_t to uint32_t
      min_iterations_u32,  // Cast size_t to uint32_t
      max_iterations_u32,  // Cast size_t to uint32_t
      num_executed_iterations_arg,
      hash_bitlen_u32,                // Cast int64_t to uint32_t
      small_hash_bitlen_u32,          // Cast size_t to uint32_t
      small_hash_reset_interval_u32,  // Cast size_t to uint32_t
      query_id_offset,                // Offset to add to query_id when calling filter
      dev_desc,
      bitset);

    last_touch.store(std::chrono::system_clock::now(), std::memory_order_relaxed);
  }

  ~persistent_runner_jit_t() noexcept override
  {
    auto whs = worker_handles.data();
    for (auto i = worker_handles.size(); i > 0; i--) {
      whs[worker_queue.pop().wait()].data.store({kNoMoreWork}, cuda::memory_order_relaxed);
    }
    RAFT_CUDA_TRY_NO_THROW(cudaStreamSynchronize(stream));
  }

  void launch(uintptr_t result_indices_ptr,
              distance_type* result_distances_ptr,
              const data_type* queries_ptr,
              uint32_t num_queries,
              uint32_t top_k)
  {
    launcher_jit_t launcher{job_queue,
                            worker_queue,
                            worker_handles.data(),
                            num_queries,
                            this->lifetime,
                            [&job_descriptors = this->job_descriptors,
                             result_indices_ptr,
                             result_distances_ptr,
                             queries_ptr,
                             top_k,
                             num_queries](uint32_t job_ix) {
                              auto& jd    = job_descriptors.data()[job_ix].input.value;
                              auto* cflag = &job_descriptors.data()[job_ix].completion_flag;
                              jd.result_indices_ptr   = result_indices_ptr;
                              jd.result_distances_ptr = result_distances_ptr;
                              jd.queries_ptr          = queries_ptr;
                              jd.top_k                = top_k;
                              jd.n_queries            = num_queries;
                              cflag->store(false, cuda::memory_order_relaxed);
                              cuda::atomic_thread_fence(cuda::memory_order_release,
                                                        cuda::thread_scope_system);
                              return cflag;
                            }};

    auto prev_touch = last_touch.load(std::memory_order_relaxed);
    if (prev_touch + lifetime / 10 < launcher.now) {
      last_touch.store(launcher.now, std::memory_order_relaxed);
    }
    launcher.wait();
  }

  auto calc_coop_grid_size(uint32_t block_size, uint32_t smem_size, float persistent_device_usage)
    -> dim3
  {
    int ctas_per_sm            = 1;
    cudaKernel_t kernel_handle = launcher->get_kernel();
    RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &ctas_per_sm, kernel_handle, block_size, smem_size));
    int num_sm    = raft::getMultiProcessorCount();
    auto n_blocks = static_cast<uint32_t>(persistent_device_usage * (ctas_per_sm * num_sm));
    if (n_blocks > kMaxWorkersNum) {
      RAFT_LOG_WARN("Limiting the grid size limit due to the size of the queue: %u -> %u",
                    n_blocks,
                    kMaxWorkersNum);
      n_blocks = kMaxWorkersNum;
    }
    return {1, n_blocks, 1};
  }
};

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterT>
void select_and_run(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  raft::device_matrix_view<const IndexT, int64_t, raft::row_major> graph,
  std::optional<raft::device_vector_view<const SourceIndexT, int64_t>> source_indices,
  uintptr_t topk_indices_ptr,     // [num_queries, topk]
  DistanceT* topk_distances_ptr,  // [num_queries, topk]
  const DataT* queries_ptr,       // [num_queries, dataset_dim]
  uint32_t num_queries,
  const IndexT* dev_seed_ptr,         // [num_queries, num_seeds]
  uint32_t* num_executed_iterations,  // [num_queries,]
  const search_params& ps,
  uint32_t topk,
  uint32_t num_itopk_candidates,
  uint32_t block_size,  //
  uint32_t smem_size,
  int64_t hash_bitlen,
  IndexT* hashmap_ptr,
  size_t small_hash_bitlen,
  size_t small_hash_reset_interval,
  uint32_t num_seeds,
  SampleFilterT sample_filter,
  cudaStream_t stream)
{
  const SourceIndexT* source_indices_ptr =
    source_indices.has_value() ? source_indices->data_handle() : nullptr;

  const auto bf = extract_cagra_sample_filter<SourceIndexT>(sample_filter);
  const cagra_bitset<SourceIndexT> bitset = bf.bitset;
  const uint32_t query_id_offset          = bf.query_id_offset;

  // Use common logic to compute launch config
  auto config             = compute_launch_config(num_itopk_candidates, ps.itopk_size, block_size);
  uint32_t max_candidates = config.max_candidates;
  uint32_t max_itopk      = config.max_itopk;
  bool topk_by_bitonic_sort               = config.topk_by_bitonic_sort;
  bool bitonic_sort_and_merge_multi_warps = config.bitonic_sort_and_merge_multi_warps;

  // Handle persistent kernels
  if (ps.persistent) {
    // Use persistent runner for JIT kernels
    using runner_type =
      persistent_runner_jit_t<DataT, IndexT, DistanceT, SourceIndexT, SampleFilterT>;

    std::string const filter_name = get_sample_filter_name<SampleFilterT>();
    std::shared_ptr<AlgorithmLauncher> launcher =
      make_cagra_single_cta_jit_launcher<DataT, IndexT, DistanceT, SourceIndexT>(
        dataset_desc,
        topk_by_bitonic_sort,
        bitonic_sort_and_merge_multi_warps,
        true /* persistent */,
        filter_name);
    if (!launcher) { RAFT_FAIL("Failed to get JIT launcher for CAGRA persistent search kernel"); }

    // Use get_runner pattern similar to non-JIT version
    const auto* dev_desc_persistent = dataset_desc.dev_ptr(stream);
    get_runner_jit<runner_type>(std::cref(dataset_desc),
                                graph,
                                source_indices_ptr,
                                max_candidates,
                                num_itopk_candidates,
                                block_size,
                                smem_size,
                                hash_bitlen,
                                small_hash_bitlen,
                                small_hash_reset_interval,
                                ps.num_random_samplings,
                                ps.rand_xor_mask,
                                num_seeds,
                                max_itopk,
                                ps.itopk_size,
                                ps.search_width,
                                ps.min_iterations,
                                ps.max_iterations,
                                sample_filter,
                                ps.persistent_lifetime,
                                ps.persistent_device_usage,
                                launcher,
                                dev_desc_persistent)  // Pass descriptor pointer
      ->launch(topk_indices_ptr, topk_distances_ptr, queries_ptr, num_queries, topk);
    return;
  } else {
    std::string const filter_name = get_sample_filter_name<SampleFilterT>();
    std::shared_ptr<AlgorithmLauncher> launcher =
      make_cagra_single_cta_jit_launcher<DataT, IndexT, DistanceT, SourceIndexT>(
        dataset_desc,
        topk_by_bitonic_sort,
        bitonic_sort_and_merge_multi_warps,
        false /* persistent */,
        filter_name);
    if (!launcher) { RAFT_FAIL("Failed to get JIT launcher for CAGRA search kernel"); }

    // Get the device descriptor pointer - dev_ptr() initializes it if needed
    const auto* dev_desc = dataset_desc.dev_ptr(stream);

    // Cast size_t/int64_t parameters to match kernel signature exactly
    // The dispatch mechanism uses void* pointers, so parameter sizes must match exactly
    const uint32_t graph_degree_u32              = static_cast<uint32_t>(graph.extent(1));
    const uint32_t hash_bitlen_u32               = static_cast<uint32_t>(hash_bitlen);
    const uint32_t small_hash_bitlen_u32         = static_cast<uint32_t>(small_hash_bitlen);
    const uint32_t small_hash_reset_interval_u32 = static_cast<uint32_t>(small_hash_reset_interval);
    const uint32_t itopk_size_u32                = static_cast<uint32_t>(ps.itopk_size);
    const uint32_t search_width_u32              = static_cast<uint32_t>(ps.search_width);
    const uint32_t min_iterations_u32            = static_cast<uint32_t>(ps.min_iterations);
    const uint32_t max_iterations_u32            = static_cast<uint32_t>(ps.max_iterations);
    const unsigned num_random_samplings_u        = static_cast<unsigned>(ps.num_random_samplings);

    dim3 grid(1, num_queries, 1);
    dim3 block(block_size, 1, 1);

    RAFT_LOG_DEBUG("Launching JIT kernel with %u threads, %u blocks, %u smem",
                   block_size,
                   num_queries,
                   smem_size);

    // Dispatch kernel via launcher
    auto kernel_launcher = [&](auto const& kernel) -> void {
      launcher->dispatch<search_single_cta_kernel_func_t<DataT, IndexT, DistanceT, SourceIndexT>>(
        stream,
        grid,
        block,
        static_cast<std::size_t>(smem_size),
        topk_indices_ptr,
        topk_distances_ptr,
        topk,
        queries_ptr,
        graph.data_handle(),
        graph_degree_u32,  // Cast int64_t to uint32_t
        source_indices_ptr,
        num_random_samplings_u,  // Cast uint32_t to unsigned for consistency
        ps.rand_xor_mask,        // uint64_t matches kernel (8 bytes)
        dev_seed_ptr,
        num_seeds,
        hashmap_ptr,
        max_candidates,
        max_itopk,
        itopk_size_u32,      // Cast size_t to uint32_t
        search_width_u32,    // Cast size_t to uint32_t
        min_iterations_u32,  // Cast size_t to uint32_t
        max_iterations_u32,  // Cast size_t to uint32_t
        num_executed_iterations,
        hash_bitlen_u32,                // Cast int64_t to uint32_t
        small_hash_bitlen_u32,          // Cast size_t to uint32_t
        small_hash_reset_interval_u32,  // Cast size_t to uint32_t
        query_id_offset,                // Offset to add to query_id when calling filter
        dev_desc,
        static_cast<IndexT>(graph.extent(0)),
        bitset);
    };

    cuvs::neighbors::detail::safely_launch_kernel_with_smem_size(
      launcher->get_kernel(), smem_size, kernel_launcher);

    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

// get_runner for JIT persistent runners (similar to non-JIT version)
template <typename RunnerT, typename... Args>
auto get_runner_jit(Args... args) -> std::shared_ptr<RunnerT>
{
  static thread_local std::weak_ptr<RunnerT> weak;
  auto runner = weak.lock();
  if (runner) {
    if (runner->param_hash == RunnerT::calculate_parameter_hash(args...)) {
      return runner;
    } else {
      weak.reset();
      runner.reset();
    }
  }
  launcher_jit_t::expected_latency = launcher_jit_t::kDefaultLatency;
  runner                           = create_runner_jit<RunnerT>(args...);
  weak                             = runner;
  return runner;
}

template <typename RunnerT, typename... Args>
auto create_runner_jit(Args... args) -> std::shared_ptr<RunnerT>
{
  std::lock_guard<std::mutex> guard(persistent.lock);
  std::shared_ptr<RunnerT> runner_outer = std::dynamic_pointer_cast<RunnerT>(persistent.runner);
  if (runner_outer) {
    // calculate_parameter_hash needs all args to match constructor signature
    // but only uses a subset for the actual hash
    if (runner_outer->param_hash == RunnerT::calculate_parameter_hash(args...)) {
      return runner_outer;
    } else {
      runner_outer.reset();
    }
  }
  persistent.runner.reset();

  cuda::std::atomic_flag ready{};
  ready.clear(cuda::std::memory_order_relaxed);
  std::thread(
    [&runner_outer, &ready](Args... thread_args) {
      runner_outer      = std::make_shared<RunnerT>(thread_args...);
      auto lifetime     = runner_outer->lifetime;
      persistent.runner = std::static_pointer_cast<persistent_runner_base_t>(runner_outer);
      std::weak_ptr<RunnerT> runner_weak = runner_outer;
      ready.test_and_set(cuda::std::memory_order_release);
      ready.notify_one();

      while (true) {
        std::this_thread::sleep_for(lifetime);
        auto runner = runner_weak.lock();
        if (!runner) { return; }
        if (runner->last_touch.load(std::memory_order_relaxed) + lifetime <
            std::chrono::system_clock::now()) {
          std::lock_guard<std::mutex> guard(persistent.lock);
          if (runner == persistent.runner) { persistent.runner.reset(); }
          return;
        }
      }
    },
    args...)
    .detach();
  ready.wait(false, cuda::std::memory_order_acquire);
  return runner_outer;
}

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search
