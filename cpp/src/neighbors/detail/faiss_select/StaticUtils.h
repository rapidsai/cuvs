/**
 * SPDX-FileCopyrightText: Copyright (c) Facebook, Inc. and its affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file thirdparty/LICENSES/LICENSE.faiss
 */

#pragma once

#include <cuda.h>
#include <raft/util/cuda_dev_essentials.cuh>

// allow usage for non-CUDA files
#ifndef __host__
#define __host__
#define __device__
#endif

namespace cuvs::neighbors::detail::faiss_select::utils {

template <typename T>
constexpr __host__ __device__ auto is_power_of2(T v) -> bool
{
  return (v && !(v & (v - 1)));
}

static_assert(is_power_of2(2048), "is_power_of2");
static_assert(!is_power_of2(3333), "is_power_of2");

template <typename T>
constexpr __host__ __device__ auto next_highest_power_of2(T v) -> T
{
  return (is_power_of2(v) ? static_cast<T>(2) * v : (static_cast<T>(1) << (raft::log2(v) + 1)));
}

static_assert(next_highest_power_of2(1) == 2, "next_highest_power_of2");
static_assert(next_highest_power_of2(2) == 4, "next_highest_power_of2");
static_assert(next_highest_power_of2(3) == 4, "next_highest_power_of2");
static_assert(next_highest_power_of2(4) == 8, "next_highest_power_of2");

static_assert(next_highest_power_of2(15) == 16, "next_highest_power_of2");
static_assert(next_highest_power_of2(16) == 32, "next_highest_power_of2");
static_assert(next_highest_power_of2(17) == 32, "next_highest_power_of2");

static_assert(next_highest_power_of2(1536000000u) == 2147483648u, "next_highest_power_of2");
static_assert(next_highest_power_of2((size_t)2147483648ULL) == (size_t)4294967296ULL,
              "next_highest_power_of2");

}  // namespace cuvs::neighbors::detail::faiss_select::utils
