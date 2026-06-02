/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace cuvs::neighbors::ivf_rabitq::detail {

// Cross-fragment device function declarations. Definitions live in their own
// JIT-LTO fragments and are resolved at nvJitLink time when a consumer planner
// adds them via add_*_device_function().
__device__ uint32_t extract_code(const uint8_t* codes, size_t d, size_t EX_BITS);

__device__ float compute_ip2_from_long_codes_warp(
  const uint8_t* vec_long_code, const float* shared_query, size_t D, size_t EX_BITS, int lane_id);

}  // namespace cuvs::neighbors::ivf_rabitq::detail
