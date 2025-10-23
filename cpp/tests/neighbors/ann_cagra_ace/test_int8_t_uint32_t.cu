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

#include <gtest/gtest.h>

#include "../ann_cagra_ace.cuh"

namespace cuvs::neighbors::cagra {

typedef AnnCagraAceTest<float, std::int8_t, std::uint32_t> AnnCagraAceTestI8_U32;
TEST_P(AnnCagraAceTestI8_U32, AnnCagraAce) { this->testAce(); }

INSTANTIATE_TEST_CASE_P(AnnCagraAceTest, AnnCagraAceTestI8_U32, ::testing::ValuesIn(ace_inputs));

}  // namespace cuvs::neighbors::cagra
