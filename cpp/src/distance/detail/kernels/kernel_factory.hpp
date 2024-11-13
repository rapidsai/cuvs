/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#pragma once

#include "gram_matrix.hpp"
#include "kernel_matrices.hpp"

namespace cuvs::distance::kernels::detail {

template <typename math_t>
class KernelFactory {
 public:
  static GramMatrixBase<math_t>* create(KernelParams params);
  [[deprecated]] static GramMatrixBase<math_t>* create(KernelParams params, cublasHandle_t handle);
};

};  // end namespace cuvs::distance::kernels::detail
