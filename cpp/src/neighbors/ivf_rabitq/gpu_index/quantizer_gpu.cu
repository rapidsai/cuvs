/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/10/25.
//

#include "quantizer_gpu.cuh"

#include <raft/core/resource/cuda_stream.hpp>

#include <atomic>
#include <thread>

namespace cuvs::neighbors::ivf_rabitq::detail {

#define MAX_D 2048

void DataQuantizerGPU::alloc_buffers(size_t num_points)
{
  const int64_t size_norm = static_cast<int64_t>(num_points) * D;
  const int64_t size_bin  = static_cast<int64_t>(num_points) * D;
  const int64_t size_xp   = static_cast<int64_t>(num_points + 1) * D;

  // Overwrite RAFT device vectors with new allocations
  d_XP_norm     = raft::make_device_vector<float, int64_t>(handle_, size_norm);
  d_bin_XP      = raft::make_device_vector<int, int64_t>(handle_, size_bin);
  d_XP          = raft::make_device_vector<float, int64_t>(handle_, size_xp);
  d_X_and_C_pad = raft::make_device_vector<float, int64_t>(handle_, size_xp);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
