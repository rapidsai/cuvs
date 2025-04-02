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

#include "../all_neighbors.cuh"

namespace cuvs::neighbors::all_neighbors {

typedef BatchANNTest<float, float> BatchANNTestF_float;
TEST_P(BatchANNTestF_float, BatchANN) { this->run(); }

INSTANTIATE_TEST_CASE_P(BatchANNTest, BatchANNTestF_float, ::testing::ValuesIn(inputsBatch));

}  // namespace cuvs::neighbors::all_neighbors
