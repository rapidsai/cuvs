/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Please add the required header files below for which panama FFM API bindings are needed

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/hnsw.h>
#include <cuvs/neighbors/ivf_pq.h>
#include <cuvs/neighbors/common.h>
#include <cuvs/neighbors/tiered_index.h>
#include <cuda_runtime.h>

/**
 * @brief function signature for setting omp threads
 */
void omp_set_num_threads(int n_writer_threads);

/**
 * @brief function signature for getting omp threads
 */
int omp_get_num_threads(void);
