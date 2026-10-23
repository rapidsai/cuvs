/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include "cub_radix_launchers.hpp"

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_ascending_RadixSortOnesweepKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_ascending::RadixSortOnesweepKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortOnesweepKernel<
    cub::detail::radix_sort::policy_selector_from_types<float, cub::NullType, unsigned long long>,
    (cub::SortOrder)0,
    float,
    cub::NullType,
    unsigned long long,
    int,
    int,
    cub::detail::identity_decomposer_t>;
}

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_ascending_RadixSortOnesweepKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_ascending::RadixSortOnesweepKernel()
{
  return &cub::detail::radix_sort::DeviceRadixSortOnesweepKernel<
    cub::detail::radix_sort::
      policy_selector_from_types<unsigned int, cub::NullType, unsigned long long>,
    (cub::SortOrder)0,
    unsigned int,
    cub::NullType,
    unsigned long long,
    int,
    int,
    cub::detail::identity_decomposer_t>;
}
