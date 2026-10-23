/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include "cub_radix_launchers.hpp"

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_ascending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_ascending::RadixSortInitLookbackKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortInitKernel<
    cub::detail::radix_sort::policy_selector_from_types<float, cub::NullType, unsigned long long>,
    int,
    int>;
}

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_descending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_descending::RadixSortInitLookbackKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortInitKernel<
    cub::detail::radix_sort::policy_selector_from_types<float, cub::NullType, unsigned long long>,
    int,
    int>;
}

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_ascending::RadixSortInitBinsAndCountersKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortInitKernel<
    cub::detail::radix_sort::policy_selector_from_types<float, cub::NullType, unsigned long long>,
    int,
    unsigned long long>;
}

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_descending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_descending::RadixSortInitBinsAndCountersKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortInitKernel<
    cub::detail::radix_sort::policy_selector_from_types<float, cub::NullType, unsigned long long>,
    int,
    unsigned long long>;
}

template <>
__host__ __device__
  cuvs::detail::radix_launch::i32_keys_u64_ascending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::i32_keys_u64_ascending::RadixSortInitLookbackKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortInitKernel<
    cub::detail::radix_sort::policy_selector_from_types<int, cub::NullType, unsigned long long>,
    int,
    int>;
}

template <>
__host__ __device__
  cuvs::detail::radix_launch::i32_keys_u64_descending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::i32_keys_u64_descending::RadixSortInitLookbackKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortInitKernel<
    cub::detail::radix_sort::policy_selector_from_types<int, cub::NullType, unsigned long long>,
    int,
    int>;
}

template <>
__host__ __device__
  cuvs::detail::radix_launch::i32_keys_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::i32_keys_u64_ascending::RadixSortInitBinsAndCountersKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortInitKernel<
    cub::detail::radix_sort::policy_selector_from_types<int, cub::NullType, unsigned long long>,
    int,
    unsigned long long>;
}

template <>
__host__ __device__
  cuvs::detail::radix_launch::i32_keys_u64_descending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::i32_keys_u64_descending::RadixSortInitBinsAndCountersKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortInitKernel<
    cub::detail::radix_sort::policy_selector_from_types<int, cub::NullType, unsigned long long>,
    int,
    unsigned long long>;
}

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_ascending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_ascending::RadixSortInitLookbackKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortInitKernel<
    cub::detail::radix_sort::
      policy_selector_from_types<unsigned int, cub::NullType, unsigned long long>,
    int,
    int>;
}

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_descending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_descending::RadixSortInitLookbackKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortInitKernel<
    cub::detail::radix_sort::
      policy_selector_from_types<unsigned int, cub::NullType, unsigned long long>,
    int,
    int>;
}

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_ascending::RadixSortInitBinsAndCountersKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortInitKernel<
    cub::detail::radix_sort::
      policy_selector_from_types<unsigned int, cub::NullType, unsigned long long>,
    int,
    unsigned long long>;
}

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_descending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_descending::RadixSortInitBinsAndCountersKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortInitKernel<
    cub::detail::radix_sort::
      policy_selector_from_types<unsigned int, cub::NullType, unsigned long long>,
    int,
    unsigned long long>;
}
