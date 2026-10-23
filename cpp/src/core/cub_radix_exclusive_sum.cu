/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include "cub_radix_launchers.hpp"

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_ascending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_ascending::RadixSortExclusiveSumKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortExclusiveSumKernel<
    cub::detail::radix_sort::policy_selector_from_types<float, cub::NullType, unsigned long long>,
    unsigned long long>;
}

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_descending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_descending::RadixSortExclusiveSumKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortExclusiveSumKernel<
    cub::detail::radix_sort::policy_selector_from_types<float, cub::NullType, unsigned long long>,
    unsigned long long>;
}

template <>
__host__ __device__
  cuvs::detail::radix_launch::i32_keys_u64_ascending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::i32_keys_u64_ascending::RadixSortExclusiveSumKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortExclusiveSumKernel<
    cub::detail::radix_sort::policy_selector_from_types<int, cub::NullType, unsigned long long>,
    unsigned long long>;
}

template <>
__host__ __device__
  cuvs::detail::radix_launch::i32_keys_u64_descending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::i32_keys_u64_descending::RadixSortExclusiveSumKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortExclusiveSumKernel<
    cub::detail::radix_sort::policy_selector_from_types<int, cub::NullType, unsigned long long>,
    unsigned long long>;
}

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_ascending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_ascending::RadixSortExclusiveSumKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortExclusiveSumKernel<
    cub::detail::radix_sort::
      policy_selector_from_types<unsigned int, cub::NullType, unsigned long long>,
    unsigned long long>;
}

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_descending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_descending::RadixSortExclusiveSumKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortExclusiveSumKernel<
    cub::detail::radix_sort::
      policy_selector_from_types<unsigned int, cub::NullType, unsigned long long>,
    unsigned long long>;
}
