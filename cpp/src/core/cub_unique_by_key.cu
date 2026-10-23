/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include "cub_unique_launchers.hpp"

template <>
__host__ __device__ cuvs::detail::unique_launch::unique_0_UniqueByKeySweepKernel_pointer
cuvs::detail::unique_launch::unique_0::UniqueByKeySweepKernel()
{
  return &cub::detail::unique_by_key::DeviceUniqueByKeySweepKernel<
    cub::detail::unique_by_key::policy_selector_from_types<unsigned int, int>,
    thrust::device_ptr<unsigned int>,
    thrust::device_ptr<int>,
    thrust::device_ptr<unsigned int>,
    thrust::device_ptr<int>,
    unsigned int*,
    cub::ScanTileState<unsigned int, true>,
    thrust::detail::wrapped_function<cuda::std::equal_to<unsigned int>, bool>,
    unsigned int>;
}

template <>
__host__ __device__ cuvs::detail::unique_launch::unique_1_UniqueByKeySweepKernel_pointer
cuvs::detail::unique_launch::unique_1::UniqueByKeySweepKernel()
{
  return &cub::detail::unique_by_key::DeviceUniqueByKeySweepKernel<
    cub::detail::unique_by_key::policy_selector_from_types<unsigned int, int>,
    thrust::device_ptr<unsigned int>,
    thrust::device_ptr<int>,
    thrust::device_ptr<unsigned int>,
    thrust::device_ptr<int>,
    unsigned long*,
    cub::ScanTileState<unsigned long long, true>,
    thrust::detail::wrapped_function<cuda::std::equal_to<unsigned int>, bool>,
    unsigned long long>;
}
