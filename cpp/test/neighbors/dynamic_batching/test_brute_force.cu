/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>

#include "../dynamic_batching.cuh"

#include <cuvs/neighbors/brute_force.hpp>

namespace cuvs::neighbors::dynamic_batching {

using brute_force_float32 = dynamic_batching_test<float,
                                                  int64_t,
                                                  brute_force::index<float, float>,
                                                  brute_force::build,
                                                  brute_force::search>;

TEST_P(brute_force_float32, defaults)
{
  build_all();
  search_all();
  check_neighbors();
}

INSTANTIATE_TEST_CASE_P(dynamic_batching, brute_force_float32, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::dynamic_batching
