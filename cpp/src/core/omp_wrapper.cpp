/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <omp.h>

#include <raft/core/logger.hpp>

#include "omp_wrapper.hpp"

namespace cuvs::core::omp {

constexpr bool is_omp_enabled()
{
#if defined(_OPENMP)
  return true;
#else
  return false;
#endif
}

int get_max_threads() { return is_omp_enabled() ? omp_get_max_threads() : 1; }
int get_num_procs() { return is_omp_enabled() ? omp_get_num_procs() : 1; }
int get_num_threads() { return is_omp_enabled() ? omp_get_num_threads() : 1; }
int get_thread_num() { return is_omp_enabled() ? omp_get_thread_num() : 0; }

void set_nested(int v)
{
  (void)v;
  if constexpr (is_omp_enabled()) { omp_set_nested(v); }
}

void check_threads(const int requirements)
{
  const int max_threads = get_max_threads();
  if (max_threads < requirements) {
    RAFT_LOG_WARN(
      "OpenMP is only allowed %d threads to run %d GPUs. Please increase the number of OpenMP "
      "threads to avoid NCCL hangs by modifying the environment variable OMP_NUM_THREADS.",
      max_threads,
      requirements);
  }
}

}  // namespace cuvs::core::omp
