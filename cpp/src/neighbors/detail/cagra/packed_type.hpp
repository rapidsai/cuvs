/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "../../ivf_pq/ivf_pq_fp_8bit.cuh"

#include <cstdint>
#include <raft/core/detail/macros.hpp>

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace cuvs::neighbors::cagra::detail::device {
template <uint32_t Bit>
struct uintN_t {};
template <>
struct uintN_t<32> {
  using type = uint32_t;
};
template <>
struct uintN_t<64> {
  using type = uint64_t;
};

template <uint32_t NumPacked, uint32_t ExpBits>
struct fp8xN {};

template <uint32_t NumPacked>
struct fp8xN<NumPacked, 5> {
  using uint_t = typename uintN_t<8 * NumPacked>::type;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using unit_t = __nv_fp8_e5m2;
  using x2_t   = __nv_fp8x2_storage_t;
#else
  using unit_t = cuvs::neighbors::ivf_pq::detail::fp_8bit<5u, true>;
#endif
  static constexpr uint32_t num_elements = NumPacked;

  union storage_t {
    unit_t x1[num_elements];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    x2_t x2[num_elements / 2];
#endif
    uint_t u;

    HDI storage_t() : u{0} {}
  } data;

  HDI fp8xN() = default;

  HDI uint_t& as_uint() { return data.u; }
  HDI uint_t as_uint() const { return data.u; }
  HDI half2 as_half2(const uint32_t i) const
  {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    return __nv_cvt_fp8x2_to_halfraw2(data.x2[i], __NV_E5M2);
#else
    half2 r;
    r.x = static_cast<half>(data.x1[2 * i]);
    r.y = static_cast<half>(data.x1[2 * i + 1]);
    return r;
#endif
  }
};
}  // namespace cuvs::neighbors::cagra::detail::device
