/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include "cub_reduce_launchers.hpp"

template <>
__host__ __device__ cuvs::detail::reduce_launch::sum_second_0_0_SingleTileSecondKernel_pointer
cuvs::detail::reduce_launch::sum_second_0_0::SingleTileSecondKernel()
{
  return &cub::detail::reduce::DeviceReduceSingleTileKernel<
    cub::detail::reduce::policy_selector_from_types<
      long,
      unsigned int,
      cuda::std::plus<long>,
      (cuda::execution::determinism::__determinism_t)1>,
    long*,
    long*,
    int,
    cuda::std::plus<long>,
    long,
    long,
    cuda::std::identity>;
}

template <>
__host__ __device__ cuvs::detail::reduce_launch::sum_second_0_1_SingleTileSecondKernel_pointer
cuvs::detail::reduce_launch::sum_second_0_1::SingleTileSecondKernel()
{
  return &cub::detail::reduce::DeviceReduceSingleTileKernel<
    cub::detail::reduce::policy_selector_from_types<
      long,
      unsigned int,
      cuda::std::plus<long>,
      (cuda::execution::determinism::__determinism_t)1>,
    long*,
    long*,
    int,
    cuda::std::plus<long>,
    long,
    long,
    cuda::std::identity>;
}

template <>
__host__ __device__ cuvs::detail::reduce_launch::sum_second_1_0_SingleTileSecondKernel_pointer
cuvs::detail::reduce_launch::sum_second_1_0::SingleTileSecondKernel()
{
  return &cub::detail::reduce::DeviceReduceSingleTileKernel<
    cub::detail::reduce::policy_selector_from_types<
      long,
      unsigned long long,
      cuda::std::plus<long>,
      (cuda::execution::determinism::__determinism_t)1>,
    long*,
    long*,
    int,
    cuda::std::plus<long>,
    long,
    long,
    cuda::std::identity>;
}

template <>
__host__ __device__ cuvs::detail::reduce_launch::sum_second_1_1_SingleTileSecondKernel_pointer
cuvs::detail::reduce_launch::sum_second_1_1::SingleTileSecondKernel()
{
  return &cub::detail::reduce::DeviceReduceSingleTileKernel<
    cub::detail::reduce::policy_selector_from_types<
      long,
      unsigned long long,
      cuda::std::plus<long>,
      (cuda::execution::determinism::__determinism_t)1>,
    long*,
    long*,
    int,
    cuda::std::plus<long>,
    long,
    long,
    cuda::std::identity>;
}
