/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <omp.h>

#include <raft/core/logger.hpp>

#include "omp_wrapper.hpp"

namespace cuvs::core::omp {

constexpr auto is_omp_enabled() -> bool
{
#if defined(_OPENMP)
  return true;
#else
  return false;
#endif
}

auto get_max_threads() -> int { return is_omp_enabled() ? omp_get_max_threads() : 1; }
auto get_num_procs() -> int { return is_omp_enabled() ? omp_get_num_procs() : 1; }
auto get_num_threads() -> int { return is_omp_enabled() ? omp_get_num_threads() : 1; }
auto get_thread_num() -> int { return is_omp_enabled() ? omp_get_thread_num() : 0; }
auto get_nested() -> int { return is_omp_enabled() ? omp_get_nested() : 0; }

void set_nested(int v)
{
  (void)v;
  if constexpr (is_omp_enabled()) { omp_set_nested(v); }
}

void set_num_threads(int v)
{
  (void)v;
  if constexpr (is_omp_enabled()) { omp_set_num_threads(v); }
}

void check_threads(const int requirements)
{
  const int max_threads = get_max_threads();
  if (max_threads < requirements) {
    RAFT_LOG_WARN(
      "Insufficient OpenMP threads: only %d available but %d required for optimal parallel "
      "execution across GPUs. Please increase OMP_NUM_THREADS to at least %d to avoid potential "
      "performance degradation or synchronization issues.",
      max_threads,
      requirements,
      requirements);
  }
}

}  // namespace cuvs::core::omp
