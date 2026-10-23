/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cub_select_launchers.hpp"
#include <cuco/detail/roaring_bitmap/roaring_bitmap_kernels.cuh>

#include <cub/device/device_select.cuh>
#include <cuda/iterator>

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::If<cuda::counting_iterator<long, __int128>,
                      long*,
                      long*,
                      cuco::experimental::detail::is_container_start<unsigned int const*>,
                      cudaStream_t,
                      0>(
  void* storage,
  size_t& bytes,
  cuda::counting_iterator<long, __int128> input,
  long* output,
  long* selected,
  cuda::std::int64_t items,
  cuco::experimental::detail::is_container_start<unsigned int const*> predicate,
  const cudaStream_t& environment);

template <>
CUB_RUNTIME_FUNCTION cudaError_t
cub::DeviceSelect::If<cuda::counting_iterator<long, __int128>,
                      long*,
                      long*,
                      cuco::experimental::detail::is_container_start<unsigned int*>,
                      cudaStream_t,
                      0>(void* storage,
                         size_t& bytes,
                         cuda::counting_iterator<long, __int128> input,
                         long* output,
                         long* selected,
                         cuda::std::int64_t items,
                         cuco::experimental::detail::is_container_start<unsigned int*> predicate,
                         const cudaStream_t& environment);
