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
constexpr __host__ __device__ auto isPowerOf2(T v) -> bool
{
  return (v && !(v & (v - 1)));
}

static_assert(isPowerOf2(2048), "isPowerOf2");
static_assert(!isPowerOf2(3333), "isPowerOf2");

template <typename T>
constexpr __host__ __device__ auto nextHighestPowerOf2(T v) -> T
{
  return (isPowerOf2(v) ? static_cast<T>(2) * v : (static_cast<T>(1) << (raft::log2(v) + 1)));
}

static_assert(nextHighestPowerOf2(1) == 2, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(2) == 4, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(3) == 4, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(4) == 8, "nextHighestPowerOf2");

static_assert(nextHighestPowerOf2(15) == 16, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(16) == 32, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(17) == 32, "nextHighestPowerOf2");

static_assert(nextHighestPowerOf2(1536000000u) == 2147483648u, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2((size_t)2147483648ULL) == (size_t)4294967296ULL,
              "nextHighestPowerOf2");

}  // namespace cuvs::neighbors::detail::faiss_select::utils
