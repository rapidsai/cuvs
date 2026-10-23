/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cub/device/device_radix_sort.cuh>

namespace cuvs::detail::radix_launch {

using f32_keys_u64_ascending = cub::detail::radix_sort::DeviceRadixSortKernelSource<
  cub::detail::radix_sort::policy_selector_from_types<float, cub::NullType, unsigned long long>,
  cub::SortOrder::Ascending,
  float,
  cub::NullType,
  unsigned long long,
  cub::detail::identity_decomposer_t>;

using f32_keys_u64_descending = cub::detail::radix_sort::DeviceRadixSortKernelSource<
  cub::detail::radix_sort::policy_selector_from_types<float, cub::NullType, unsigned long long>,
  cub::SortOrder::Descending,
  float,
  cub::NullType,
  unsigned long long,
  cub::detail::identity_decomposer_t>;

using i32_keys_u64_ascending = cub::detail::radix_sort::DeviceRadixSortKernelSource<
  cub::detail::radix_sort::policy_selector_from_types<int, cub::NullType, unsigned long long>,
  cub::SortOrder::Ascending,
  int,
  cub::NullType,
  unsigned long long,
  cub::detail::identity_decomposer_t>;

using i32_keys_u64_descending = cub::detail::radix_sort::DeviceRadixSortKernelSource<
  cub::detail::radix_sort::policy_selector_from_types<int, cub::NullType, unsigned long long>,
  cub::SortOrder::Descending,
  int,
  cub::NullType,
  unsigned long long,
  cub::detail::identity_decomposer_t>;

using f32_keys_u64_ascending_RadixSortExclusiveSumKernel_pointer = void (*)(unsigned long long*);

using f32_keys_u64_descending_RadixSortExclusiveSumKernel_pointer = void (*)(unsigned long long*);

using i32_keys_u64_ascending_RadixSortExclusiveSumKernel_pointer = void (*)(unsigned long long*);

using i32_keys_u64_descending_RadixSortExclusiveSumKernel_pointer = void (*)(unsigned long long*);

using f32_keys_u64_ascending_RadixSortHistogramKernel_pointer =
  void (*)(unsigned long long*,
           float const*,
           unsigned long long,
           int,
           int,
           cub::detail::identity_decomposer_t);

using f32_keys_u64_ascending_RadixSortInitLookbackKernel_pointer = void (*)(int*,
                                                                            unsigned long,
                                                                            int*,
                                                                            unsigned long);

using f32_keys_u64_descending_RadixSortInitLookbackKernel_pointer = void (*)(int*,
                                                                             unsigned long,
                                                                             int*,
                                                                             unsigned long);

using f32_keys_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer =
  void (*)(int*, unsigned long, unsigned long long*, unsigned long);

using f32_keys_u64_descending_RadixSortInitBinsAndCountersKernel_pointer =
  void (*)(int*, unsigned long, unsigned long long*, unsigned long);

using i32_keys_u64_ascending_RadixSortInitLookbackKernel_pointer = void (*)(int*,
                                                                            unsigned long,
                                                                            int*,
                                                                            unsigned long);

using i32_keys_u64_descending_RadixSortInitLookbackKernel_pointer = void (*)(int*,
                                                                             unsigned long,
                                                                             int*,
                                                                             unsigned long);

using i32_keys_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer =
  void (*)(int*, unsigned long, unsigned long long*, unsigned long);

using i32_keys_u64_descending_RadixSortInitBinsAndCountersKernel_pointer =
  void (*)(int*, unsigned long, unsigned long long*, unsigned long);

using f32_keys_u64_ascending_RadixSortOnesweepKernel_pointer =
  void (*)(int*,
           int*,
           unsigned long long*,
           unsigned long long const*,
           float*,
           float const*,
           cub::NullType*,
           cub::NullType const*,
           int,
           int,
           int,
           cub::detail::identity_decomposer_t);

using f32_keys_u64_ascending_RadixSortSingleTileKernel_pointer =
  void (*)(float const*,
           float*,
           cub::NullType const*,
           cub::NullType*,
           unsigned long long,
           int,
           int,
           cub::detail::identity_decomposer_t);

}  // namespace cuvs::detail::radix_launch

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_ascending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_ascending::RadixSortExclusiveSumKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_descending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_descending::RadixSortExclusiveSumKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::i32_keys_u64_ascending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::i32_keys_u64_ascending::RadixSortExclusiveSumKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::i32_keys_u64_descending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::i32_keys_u64_descending::RadixSortExclusiveSumKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_ascending_RadixSortHistogramKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_ascending::RadixSortHistogramKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_ascending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_ascending::RadixSortInitLookbackKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_descending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_descending::RadixSortInitLookbackKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_ascending::RadixSortInitBinsAndCountersKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_descending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_descending::RadixSortInitBinsAndCountersKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::i32_keys_u64_ascending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::i32_keys_u64_ascending::RadixSortInitLookbackKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::i32_keys_u64_descending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::i32_keys_u64_descending::RadixSortInitLookbackKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::i32_keys_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::i32_keys_u64_ascending::RadixSortInitBinsAndCountersKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::i32_keys_u64_descending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::i32_keys_u64_descending::RadixSortInitBinsAndCountersKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_ascending_RadixSortOnesweepKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_ascending::RadixSortOnesweepKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_ascending_RadixSortSingleTileKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_ascending::RadixSortSingleTileKernel();

namespace cuvs::detail::radix_launch {
using u32_keys_u64_ascending = cub::detail::radix_sort::DeviceRadixSortKernelSource<
  cub::detail::radix_sort::
    policy_selector_from_types<unsigned int, cub::NullType, unsigned long long>,
  cub::SortOrder::Ascending,
  unsigned int,
  cub::NullType,
  unsigned long long,
  cub::detail::identity_decomposer_t>;
using u32_keys_u64_descending = cub::detail::radix_sort::DeviceRadixSortKernelSource<
  cub::detail::radix_sort::
    policy_selector_from_types<unsigned int, cub::NullType, unsigned long long>,
  cub::SortOrder::Descending,
  unsigned int,
  cub::NullType,
  unsigned long long,
  cub::detail::identity_decomposer_t>;
using u32_keys_u64_ascending_RadixSortExclusiveSumKernel_pointer  = void (*)(unsigned long long*);
using u32_keys_u64_descending_RadixSortExclusiveSumKernel_pointer = void (*)(unsigned long long*);
using u32_keys_u64_ascending_RadixSortHistogramKernel_pointer =
  void (*)(unsigned long long*,
           unsigned int const*,
           unsigned long long,
           int,
           int,
           cub::detail::identity_decomposer_t);
using u32_keys_u64_ascending_RadixSortInitLookbackKernel_pointer  = void (*)(int*,
                                                                            unsigned long,
                                                                            int*,
                                                                            unsigned long);
using u32_keys_u64_descending_RadixSortInitLookbackKernel_pointer = void (*)(int*,
                                                                             unsigned long,
                                                                             int*,
                                                                             unsigned long);
using u32_keys_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer =
  void (*)(int*, unsigned long, unsigned long long*, unsigned long);
using u32_keys_u64_descending_RadixSortInitBinsAndCountersKernel_pointer =
  void (*)(int*, unsigned long, unsigned long long*, unsigned long);
using u32_keys_u64_ascending_RadixSortOnesweepKernel_pointer =
  void (*)(int*,
           int*,
           unsigned long long*,
           unsigned long long const*,
           unsigned int*,
           unsigned int const*,
           cub::NullType*,
           cub::NullType const*,
           int,
           int,
           int,
           cub::detail::identity_decomposer_t);
using u32_keys_u64_ascending_RadixSortSingleTileKernel_pointer =
  void (*)(unsigned int const*,
           unsigned int*,
           cub::NullType const*,
           cub::NullType*,
           unsigned long long,
           int,
           int,
           cub::detail::identity_decomposer_t);
}  // namespace cuvs::detail::radix_launch

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_ascending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_ascending::RadixSortExclusiveSumKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_descending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_descending::RadixSortExclusiveSumKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_ascending_RadixSortHistogramKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_ascending::RadixSortHistogramKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_ascending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_ascending::RadixSortInitLookbackKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_descending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_descending::RadixSortInitLookbackKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_ascending::RadixSortInitBinsAndCountersKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_descending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_descending::RadixSortInitBinsAndCountersKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_ascending_RadixSortOnesweepKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_ascending::RadixSortOnesweepKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_ascending_RadixSortSingleTileKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_ascending::RadixSortSingleTileKernel();
