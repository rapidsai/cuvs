/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include "../ann_scann.cuh"

namespace cuvs::neighbors::experimental::scann {

using f32_i64 = scann_test<float, int64_t>;

TEST_BUILD(f32_i64)
TEST_BUILD_HOST_INPUT(f32_i64)
TEST_BUILD_HOST_INPUT_OVERLAP(f32_i64);

INSTANTIATE(f32_i64,
            defaults() + small_dims_all_pq_bits() + big_dims_all_pq_bits() + bf16() + avq() +
              soar());

}  // namespace cuvs::neighbors::experimental::scann
