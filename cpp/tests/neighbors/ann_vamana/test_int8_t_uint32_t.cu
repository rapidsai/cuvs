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

#include "../ann_vamana.cuh"

namespace cuvs::neighbors::experimental::vamana {

typedef AnnVamanaTest<float, int8_t, std::uint32_t> AnnVamanaTestI8_U32;
TEST_P(AnnVamanaTestI8_U32, AnnVamana) { this->testVamana(); }

INSTANTIATE_TEST_CASE_P(AnnVamanaTest, AnnVamanaTestI8_U32, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::experimental::vamana
