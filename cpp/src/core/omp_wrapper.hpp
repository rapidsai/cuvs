/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <omp.h>

namespace cuvs::core::omp {

constexpr bool is_omp_enabled();

int get_max_threads();
int get_num_procs();
int get_num_threads();
int get_thread_num();
int get_nested();

void set_nested(int v);
void set_num_threads(int v);

void check_threads(const int requirements);

}  // namespace cuvs::core::omp
