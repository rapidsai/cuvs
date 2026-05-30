/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdlib>
#include <cstring>

namespace cuvs::neighbors::ivf_rabitq::detail {

// Transitional A/B switch for the JIT-LTO refactor of the IVF-RaBitQ search kernels.
// Set CUVS_IVF_RABITQ_USE_JIT_LTO=1 to route through the JIT-LTO launcher path.
// Default (unset or any other value): legacy `__global__` kernel-pointer dispatch.
//
// The env var is read once on first call and cached for the process lifetime.
// TODO: remove once the JIT-LTO refactor is complete and the legacy path is dropped.
inline bool use_jit_lto_search()
{
  static bool const value = []() {
    const char* env = std::getenv("CUVS_IVF_RABITQ_USE_JIT_LTO");
    return env != nullptr && std::strcmp(env, "1") == 0;
  }();
  return value;
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
