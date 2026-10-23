/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <thrust/device_ptr.h>

#include <cub/device/device_reduce.cuh>
#include <thrust/reduce.h>

namespace cuvs::detail::reduce_launch {
using sum_second_0_0 = cub::detail::reduce::DeviceReduceKernelSource<
  cub::detail::reduce::policy_selector_from_types<long,
                                                  unsigned int,
                                                  cuda::std::plus<long>,
                                                  (cuda::execution::determinism::__determinism_t)1>,
  thrust::device_ptr<long>,
  long*,
  unsigned int,
  unsigned int,
  cuda::std::plus<long>,
  long,
  long,
  cuda::std::identity,
  true>;
using sum_second_0_1 = cub::detail::reduce::DeviceReduceKernelSource<
  cub::detail::reduce::policy_selector_from_types<long,
                                                  unsigned int,
                                                  cuda::std::plus<long>,
                                                  (cuda::execution::determinism::__determinism_t)1>,
  long*,
  long*,
  unsigned int,
  unsigned int,
  cuda::std::plus<long>,
  long,
  long,
  cuda::std::identity,
  true>;
using sum_second_1_0 = cub::detail::reduce::DeviceReduceKernelSource<
  cub::detail::reduce::policy_selector_from_types<long,
                                                  unsigned long long,
                                                  cuda::std::plus<long>,
                                                  (cuda::execution::determinism::__determinism_t)1>,
  thrust::device_ptr<long>,
  long*,
  unsigned long long,
  unsigned long long,
  cuda::std::plus<long>,
  long,
  long,
  cuda::std::identity,
  true>;
using sum_second_1_1 = cub::detail::reduce::DeviceReduceKernelSource<
  cub::detail::reduce::policy_selector_from_types<long,
                                                  unsigned long long,
                                                  cuda::std::plus<long>,
                                                  (cuda::execution::determinism::__determinism_t)1>,
  long*,
  long*,
  unsigned long long,
  unsigned long long,
  cuda::std::plus<long>,
  long,
  long,
  cuda::std::identity,
  true>;

using sum_second_0_0_SingleTileSecondKernel_pointer =
  void (*)(long*, long*, int, cuda::std::plus<long>, long, cuda::std::identity);
using sum_second_0_1_SingleTileSecondKernel_pointer =
  void (*)(long*, long*, int, cuda::std::plus<long>, long, cuda::std::identity);
using sum_second_1_0_SingleTileSecondKernel_pointer =
  void (*)(long*, long*, int, cuda::std::plus<long>, long, cuda::std::identity);
using sum_second_1_1_SingleTileSecondKernel_pointer =
  void (*)(long*, long*, int, cuda::std::plus<long>, long, cuda::std::identity);
}  // namespace cuvs::detail::reduce_launch

template <>
__host__ __device__ cuvs::detail::reduce_launch::sum_second_0_0_SingleTileSecondKernel_pointer
cuvs::detail::reduce_launch::sum_second_0_0::SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::reduce_launch::sum_second_0_1_SingleTileSecondKernel_pointer
cuvs::detail::reduce_launch::sum_second_0_1::SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::reduce_launch::sum_second_1_0_SingleTileSecondKernel_pointer
cuvs::detail::reduce_launch::sum_second_1_0::SingleTileSecondKernel();

template <>
__host__ __device__ cuvs::detail::reduce_launch::sum_second_1_1_SingleTileSecondKernel_pointer
cuvs::detail::reduce_launch::sum_second_1_1::SingleTileSecondKernel();
