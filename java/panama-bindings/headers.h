/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/all.h>

// Please add any extra required header files below for which panama FFM API bindings are needed
#include <cuda_runtime.h>

/**
 * @brief function signature for setting omp threads
 */
void omp_set_num_threads(int n_writer_threads);

/**
 * @brief function signature for getting omp threads
 */
int omp_get_num_threads(void);
