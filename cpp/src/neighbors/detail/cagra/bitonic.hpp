/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// TODO: This shouldn't be calling RAFT detail APIs
#include <raft/core/detail/macros.hpp>

#include <cstdint>

namespace cuvs::neighbors::cagra::detail {
namespace bitonic {

namespace detail {

template <class K, class V>
RAFT_DEVICE_INLINE_FUNCTION void swap_if_needed(K& k0, V& v0, K& k1, V& v1, const bool asc)
{
  if ((k0 != k1) && ((k0 < k1) != asc)) {
    const auto tmp_k = k0;
    k0               = k1;
    k1               = tmp_k;
    const auto tmp_v = v0;
    v0               = v1;
    v1               = tmp_v;
  }
}

template <class K, class V>
RAFT_DEVICE_INLINE_FUNCTION void swap_if_needed(K& k0,
                                                V& v0,
                                                const unsigned lane_offset,
                                                const bool asc)
{
  auto k1 = __shfl_xor_sync(~0u, k0, lane_offset);
  auto v1 = __shfl_xor_sync(~0u, v0, lane_offset);
  if ((k0 != k1) && ((k0 < k1) != asc)) {
    k0 = k1;
    v0 = v1;
  }
}

template <class K, class V, unsigned warp_size = 32>
struct warp_merge_core_n {
  RAFT_DEVICE_INLINE_FUNCTION void operator()(
    K* ks, V* vs, unsigned n, const std::uint32_t range, const bool asc)
  {
    const auto lane_id = threadIdx.x % warp_size;

    if (range == 1) {
      for (std::uint32_t b = 2; b <= n; b <<= 1) {
        for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
#pragma unroll
          for (std::uint32_t i = 0; i < n; i++) {
            std::uint32_t j = i ^ c;
            if (i >= j) continue;
            const auto line_id = i + (n * lane_id);
            const auto p       = static_cast<bool>(line_id & b) == static_cast<bool>(line_id & c);
            swap_if_needed(ks[i], vs[i], ks[j], vs[j], p);
          }
        }
      }
    } else {
      const std::uint32_t b = range;
      for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
        const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
#pragma unroll
        for (std::uint32_t i = 0; i < n; i++) {
          swap_if_needed(ks[i], vs[i], c, p);
        }
      }
      const auto p = ((lane_id & b) == 0);
      for (std::uint32_t c = n / 2; c >= 1; c >>= 1) {
#pragma unroll
        for (std::uint32_t i = 0; i < n; i++) {
          std::uint32_t j = i ^ c;
          if (i >= j) continue;
          swap_if_needed(ks[i], vs[i], ks[j], vs[j], p);
        }
      }
    }
  }
};

template <class K, class V, unsigned warp_size>
struct warp_merge_core_2 {
  RAFT_DEVICE_INLINE_FUNCTION void operator()(K* ks,
                                              V* vs,
                                              const std::uint32_t range,
                                              const bool asc)
  {
    constexpr unsigned N = 2;
    const auto lane_id   = threadIdx.x % warp_size;

    if (range == 1) {
      const auto p = ((lane_id & 1) == 0);
      swap_if_needed(ks[0], vs[0], ks[1], vs[1], p);
    } else {
      const std::uint32_t b = range;
      for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
        const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
#pragma unroll
        for (std::uint32_t i = 0; i < N; i++) {
          swap_if_needed(ks[i], vs[i], c, p);
        }
      }
      const auto p = ((lane_id & b) == 0);
      swap_if_needed(ks[0], vs[0], ks[1], vs[1], p);
    }
  }
};

template <class K, class V, unsigned warp_size>
struct warp_merge_core_1 {
  RAFT_DEVICE_INLINE_FUNCTION void operator()(K* ks,
                                              V* vs,
                                              const std::uint32_t range,
                                              const bool asc)
  {
    const auto lane_id    = threadIdx.x % warp_size;
    const std::uint32_t b = range;
    for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
      const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
      swap_if_needed(ks[0], vs[0], c, p);
    }
  }
};

}  // namespace detail

template <class K, class V, unsigned warp_size = 32>
RAFT_DEVICE_INLINE_FUNCTION void warp_merge(
  K* ks, V* vs, unsigned n, unsigned range, const bool asc = true)
{
  if (n == 1) {
    detail::warp_merge_core_1<K, V, warp_size>{}(ks, vs, range, asc);
  } else if (n == 2) {
    detail::warp_merge_core_2<K, V, warp_size>{}(ks, vs, range, asc);
  } else {
    detail::warp_merge_core_n<K, V, warp_size>{}(ks, vs, n, range, asc);
  }
}

template <class K, class V, unsigned warp_size = 32>
RAFT_DEVICE_INLINE_FUNCTION void warp_sort(K* ks, V* vs, unsigned n, const bool asc = true)
{
#pragma unroll 1
  for (std::uint32_t range = 1; range <= warp_size; range <<= 1) {
    warp_merge<K, V, warp_size>(ks, vs, n, range, asc);
  }
}

}  // namespace bitonic
}  // namespace cuvs::neighbors::cagra::detail
