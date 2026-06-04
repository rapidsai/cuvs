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
    std::mutex mutex;

    explicit state(PayloadT payload) : host_payload(payload) {}

    ~state() noexcept
    {
      if (device_payload != nullptr) {
        RAFT_CUDA_TRY_NO_THROW(cudaFreeAsync(device_payload, stream));
      }
    }

    PayloadT* dev_ptr(cudaStream_t cuda_stream)
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (device_payload == nullptr) {
        RAFT_CUDA_TRY(
          cudaMallocAsync(reinterpret_cast<void**>(&device_payload), sizeof(PayloadT), cuda_stream));
        RAFT_CUDA_TRY(cudaMemcpyAsync(device_payload,
                                      &host_payload,
                                      sizeof(PayloadT),
                                      cudaMemcpyHostToDevice,
                                      cuda_stream));
        stream = cuda_stream;
      }
      return device_payload;
    }
  };

  cagra_device_payload_owner() = default;

  explicit cagra_device_payload_owner(PayloadT payload)
  {
    static std::mutex cache_mutex;
    static std::unordered_map<std::uint64_t, std::shared_ptr<state>> cache;

    const auto key = cagra_payload_hash(payload);
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (auto it = cache.find(key); it != cache.end()) {
      if (auto cached = it->second;
          std::memcmp(&cached->host_payload, &payload, sizeof(PayloadT)) == 0) {
        state_ = std::move(cached);
        return;
      }
    }
    state_    = std::make_shared<state>(payload);
    cache[key] = state_;
  }

  void* dev_ptr(cudaStream_t stream) const
  {
    return state_ == nullptr ? nullptr : state_->dev_ptr(stream);
  }

  PayloadT const* host_payload() const
  {
    return state_ == nullptr ? nullptr : &state_->host_payload;
  }

 private:
  std::shared_ptr<state> state_;
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
