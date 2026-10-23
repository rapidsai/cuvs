/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_ptx_query.hpp"

#include "cub_scan_launchers.hpp"

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i32_u32_exclusive_pointer_ScanKernel_pointer
cuvs::detail::scan_launch::sum_i32_u32_exclusive_pointer::ScanKernel()
{
  return &cub::detail::scan::DeviceScanKernel<
    cub::detail::scan::
      policy_selector_from_types<int*, int*, int, unsigned int, cuda::std::plus<void>, false>,
    int*,
    int*,
    cub::ScanTileState<int, true>,
    cuda::std::plus<void>,
    cub::detail::InputValue<int, int*>,
    unsigned int,
    int,
    false,
    false,
    int>;
}

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i32_ulong_exclusive_pointer_ScanKernel_pointer
cuvs::detail::scan_launch::sum_i32_ulong_exclusive_pointer::ScanKernel()
{
  return &cub::detail::scan::DeviceScanKernel<
    cub::detail::scan::
      policy_selector_from_types<int*, int*, int, unsigned long, cuda::std::plus<void>, false>,
    int*,
    int*,
    cub::ScanTileState<int, true>,
    cuda::std::plus<void>,
    cub::detail::InputValue<int, int*>,
    unsigned long,
    int,
    false,
    false,
    int>;
}

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i64_u32_exclusive_pointer_ScanKernel_pointer
cuvs::detail::scan_launch::sum_i64_u32_exclusive_pointer::ScanKernel()
{
  return &cub::detail::scan::DeviceScanKernel<
    cub::detail::scan::
      policy_selector_from_types<long*, long*, long, unsigned int, cuda::std::plus<void>, false>,
    long*,
    long*,
    cub::ScanTileState<long, true>,
    cuda::std::plus<void>,
    cub::detail::InputValue<long, long*>,
    unsigned int,
    long,
    false,
    false,
    long>;
}

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i64_u64_exclusive_pointer_ScanKernel_pointer
cuvs::detail::scan_launch::sum_i64_u64_exclusive_pointer::ScanKernel()
{
  return &cub::detail::scan::DeviceScanKernel<
    cub::detail::scan::policy_selector_from_types<long*,
                                                  long*,
                                                  long,
                                                  unsigned long long,
                                                  cuda::std::plus<void>,
                                                  false>,
    long*,
    long*,
    cub::ScanTileState<long, true>,
    cuda::std::plus<void>,
    cub::detail::InputValue<long, long*>,
    unsigned long long,
    long,
    false,
    false,
    long>;
}

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i32_u32_exclusive_device_ScanKernel_pointer
cuvs::detail::scan_launch::sum_i32_u32_exclusive_device::ScanKernel()
{
  return &cub::detail::scan::DeviceScanKernel<
    cub::detail::scan::policy_selector_from_types<thrust::device_ptr<int>,
                                                  thrust::device_ptr<int>,
                                                  int,
                                                  unsigned int,
                                                  cuda::std::plus<void>,
                                                  false>,
    int*,
    int*,
    cub::ScanTileState<int, true>,
    cuda::std::plus<void>,
    cub::detail::InputValue<int, int*>,
    unsigned int,
    int,
    false,
    false,
    int>;
}

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i32_ulong_exclusive_device_ScanKernel_pointer
cuvs::detail::scan_launch::sum_i32_ulong_exclusive_device::ScanKernel()
{
  return &cub::detail::scan::DeviceScanKernel<
    cub::detail::scan::policy_selector_from_types<thrust::device_ptr<int>,
                                                  thrust::device_ptr<int>,
                                                  int,
                                                  unsigned long,
                                                  cuda::std::plus<void>,
                                                  false>,
    int*,
    int*,
    cub::ScanTileState<int, true>,
    cuda::std::plus<void>,
    cub::detail::InputValue<int, int*>,
    unsigned long,
    int,
    false,
    false,
    int>;
}

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i64_u32_exclusive_device_ScanKernel_pointer
cuvs::detail::scan_launch::sum_i64_u32_exclusive_device::ScanKernel()
{
  return &cub::detail::scan::DeviceScanKernel<
    cub::detail::scan::policy_selector_from_types<thrust::device_ptr<long>,
                                                  thrust::device_ptr<long>,
                                                  long,
                                                  unsigned int,
                                                  cuda::std::plus<void>,
                                                  false>,
    long*,
    long*,
    cub::ScanTileState<long, true>,
    cuda::std::plus<void>,
    cub::detail::InputValue<long, long*>,
    unsigned int,
    long,
    false,
    false,
    long>;
}

template <>
__host__ __device__ cuvs::detail::scan_launch::sum_i64_ulong_exclusive_device_ScanKernel_pointer
cuvs::detail::scan_launch::sum_i64_ulong_exclusive_device::ScanKernel()
{
  return &cub::detail::scan::DeviceScanKernel<
    cub::detail::scan::policy_selector_from_types<thrust::device_ptr<long>,
                                                  thrust::device_ptr<long>,
                                                  long,
                                                  unsigned long,
                                                  cuda::std::plus<void>,
                                                  false>,
    long*,
    long*,
    cub::ScanTileState<long, true>,
    cuda::std::plus<void>,
    cub::detail::InputValue<long, long*>,
    unsigned long,
    long,
    false,
    false,
    long>;
}
