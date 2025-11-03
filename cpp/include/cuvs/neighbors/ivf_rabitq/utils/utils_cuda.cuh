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

//
// Created by Stardust on 3/4/25.
//

#include <iostream>

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " " \
                << cudaGetErrorString(err) << std::endl;                  \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  } while (0)

#define RABITQ_CUBLAS_CHECK(expr)                                                             \
  do {                                                                                        \
    cublasStatus_t _st = (expr);                                                              \
    if (_st != CUBLAS_STATUS_SUCCESS) {                                                       \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " << _st << "\n"; \
      std::exit(EXIT_FAILURE);                                                                \
    }                                                                                         \
  } while (0)
