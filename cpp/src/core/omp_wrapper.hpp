/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <omp.h>

namespace cuvs::core::omp {

constexpr auto is_omp_enabled() -> bool;

auto get_max_threads() -> int;
auto get_num_procs() -> int;
auto get_num_threads() -> int;
auto get_thread_num() -> int;
auto get_nested() -> int;

void set_nested(int v);
void set_num_threads(int v);

void check_threads(const int requirements);

}  // namespace cuvs::core::omp
