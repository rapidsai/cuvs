/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../../sample_filter.cuh"  // public filter types
#include "../../sample_filter_data.cuh"

#if !defined(__CUDACC_RTC__)
#include <raft/core/error.hpp>

#include <cuda_runtime_api.h>
#endif

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if !defined(__CUDACC_RTC__)
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#endif

namespace cuvs::neighbors::cagra::detail {

template <typename SourceIndexT>
using cagra_filter_data_storage = ::cuvs::neighbors::detail::bitset_filter_data_t<SourceIndexT>;

/// Device payload for linked CAGRA sample filters plus query offset for wrapped filters.
template <typename SourceIndexT>
struct cagra_sample_filter {
  void* filter_data{nullptr};
  std::uint32_t query_id_offset{0};

  __device__ __forceinline__ void* sample_filter_data() { return filter_data; }
};

#if !defined(__CUDACC_RTC__)

template <typename PayloadT>
std::uint64_t cagra_payload_hash(PayloadT const& payload)
{
  static_assert(std::is_trivially_copyable_v<PayloadT>);
  constexpr std::uint64_t kOffset = 1469598103934665603ull;
  constexpr std::uint64_t kPrime  = 1099511628211ull;
  auto const* bytes               = reinterpret_cast<unsigned char const*>(&payload);
  std::uint64_t hash              = kOffset;
  for (std::size_t i = 0; i < sizeof(PayloadT); ++i) {
    hash ^= bytes[i];
    hash *= kPrime;
  }
  return hash;
}

template <typename PayloadT>
struct cagra_device_payload_owner {
  struct state {
    PayloadT host_payload{};
    PayloadT* device_payload{nullptr};
    cudaStream_t stream{};
    cudaEvent_t ready_event{};
    int device{-1};
    std::mutex mutex;

    explicit state(PayloadT payload) : host_payload(payload) {}

    ~state() noexcept
    {
      if (device_payload != nullptr) {
        RAFT_CUDA_TRY_NO_THROW(cudaFreeAsync(device_payload, stream));
      }
      if (ready_event != nullptr) { RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(ready_event)); }
    }

    PayloadT* dev_ptr(cudaStream_t cuda_stream)
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (device_payload == nullptr) {
        RAFT_CUDA_TRY(cudaGetDevice(&device));
        RAFT_CUDA_TRY(
          cudaMallocAsync(reinterpret_cast<void**>(&device_payload), sizeof(PayloadT), cuda_stream));
        RAFT_CUDA_TRY(cudaMemcpyAsync(device_payload,
                                      &host_payload,
                                      sizeof(PayloadT),
                                      cudaMemcpyHostToDevice,
                                      cuda_stream));
        RAFT_CUDA_TRY(cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming));
        RAFT_CUDA_TRY(cudaEventRecord(ready_event, cuda_stream));
        stream = cuda_stream;
      } else {
        RAFT_CUDA_TRY(cudaStreamWaitEvent(cuda_stream, ready_event, 0));
      }
      return device_payload;
    }
  };

  struct cache_key {
    std::uint64_t payload_hash{};
    int device{};

    bool operator==(cache_key const& other) const
    {
      return payload_hash == other.payload_hash && device == other.device;
    }
  };

  struct cache_key_hash {
    std::size_t operator()(cache_key const& key) const
    {
      auto seed = static_cast<std::size_t>(key.payload_hash);
      seed ^= static_cast<std::size_t>(key.device) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      return seed;
    }
  };

  cagra_device_payload_owner() = default;

  explicit cagra_device_payload_owner(PayloadT payload)
    : state_{std::make_shared<state>(payload)}
  {
  }

  void* dev_ptr(cudaStream_t stream) const
  {
    if (state_ == nullptr) { return nullptr; }

    int device{};
    RAFT_CUDA_TRY(cudaGetDevice(&device));

    static std::mutex cache_mutex;
    // Keep cached payload copies for process lifetime to avoid per-search allocation/copy churn.
    // Cross-stream reuse is ordered by each state's ready_event before kernels consume the pointer.
    static std::unordered_map<cache_key, std::vector<std::shared_ptr<state>>, cache_key_hash> cache;

    const auto key = cache_key{cagra_payload_hash(state_->host_payload), device};
    std::shared_ptr<state> selected_state;
    {
      std::lock_guard<std::mutex> lock(cache_mutex);
      auto& entries = cache[key];
      for (auto const& cached : entries) {
        if (std::memcmp(&cached->host_payload, &state_->host_payload, sizeof(PayloadT)) == 0) {
          selected_state = cached;
          break;
        }
      }
      if (selected_state == nullptr) {
        selected_state = state_;
        entries.push_back(selected_state);
      }
    }

    state_ = std::move(selected_state);
    return state_->dev_ptr(stream);
  }

  PayloadT const* host_payload() const
  {
    return state_ == nullptr ? nullptr : &state_->host_payload;
  }

 private:
  mutable std::shared_ptr<state> state_;
};

template <typename SourceIndexT>
struct cagra_sample_filter_payload {
  cagra_sample_filter<SourceIndexT> payload{};
  cagra_device_payload_owner<cagra_filter_data_storage<SourceIndexT>> storage_owner{};

  cagra_sample_filter<SourceIndexT> device_payload(cudaStream_t stream) const
  {
    auto out = payload;
    if (out.filter_data == nullptr) { out.filter_data = storage_owner.dev_ptr(stream); }
    return out;
  }
};

template <typename T>
struct is_bitset_filter : std::false_type {};

template <typename bitset_t, typename index_t>
struct is_bitset_filter<::cuvs::neighbors::filtering::bitset_filter<bitset_t, index_t>>
  : std::true_type {};

template <typename T>
struct is_udf_filter : std::false_type {};

template <>
struct is_udf_filter<::cuvs::neighbors::filtering::udf_filter> : std::true_type {};

template <typename SourceIndexT, typename FilterT>
void fill_cagra_sample_filter(cagra_sample_filter_payload<SourceIndexT>& out, const FilterT& filter)
{
  using DecayedFilter = std::decay_t<FilterT>;
  if constexpr (is_bitset_filter<DecayedFilter>::value) {
    const auto bitset_view = filter.view();
    out.storage_owner =
      cagra_device_payload_owner<cagra_filter_data_storage<SourceIndexT>>{
        cagra_filter_data_storage<SourceIndexT>{const_cast<std::uint32_t*>(bitset_view.data()),
                                                static_cast<SourceIndexT>(bitset_view.size()),
                                                static_cast<SourceIndexT>(
                                                  bitset_view.get_original_nbits())}};
  } else if constexpr (is_udf_filter<DecayedFilter>::value) {
    out.payload.filter_data = filter.filter_data;
  }
}

/// Host: fill @ref cagra_sample_filter_payload from a CAGRA filter object.
template <typename SourceIndexT, typename SampleFilterT>
cagra_sample_filter_payload<SourceIndexT> extract_cagra_sample_filter(
  const SampleFilterT& sample_filter)
{
  cagra_sample_filter_payload<SourceIndexT> out;
  if constexpr (requires {
                  sample_filter.filter;
                  sample_filter.offset;
                }) {
    out.payload.query_id_offset = sample_filter.offset;
    fill_cagra_sample_filter(out, sample_filter.filter);
  } else {
    fill_cagra_sample_filter(out, sample_filter);
  }
  return out;
}

/// Host: find UDF compile/link metadata only. Query offsets stay in the runtime payload produced
/// by @ref extract_cagra_sample_filter and are applied before calling the linked sample_filter.
template <typename SampleFilterT>
const ::cuvs::neighbors::filtering::udf_filter* get_cagra_udf_filter(
  const SampleFilterT& sample_filter)
{
  using DecayedFilter = std::decay_t<SampleFilterT>;
  if constexpr (is_udf_filter<DecayedFilter>::value) {
    return &sample_filter;
  } else if constexpr (requires { sample_filter.filter; }) {
    return get_cagra_udf_filter(sample_filter.filter);
  } else {
    return nullptr;
  }
}

#endif  // !defined(__CUDACC_RTC__)

}  // namespace cuvs::neighbors::cagra::detail
