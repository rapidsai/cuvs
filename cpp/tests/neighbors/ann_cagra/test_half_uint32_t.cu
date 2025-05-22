/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

typedef AnnCagraTest<float, half, std::uint32_t> AnnCagraTestF16_U32;
TEST_P(AnnCagraTestF16_U32, AnnCagra_U32) { this->testCagra<uint32_t>(); }
TEST_P(AnnCagraTestF16_U32, AnnCagra_I64) { this->testCagra<int64_t>(); }

typedef AnnCagraIndexMergeTest<float, half, std::uint32_t> AnnCagraIndexMergeTestF16_U32;
TEST_P(AnnCagraIndexMergeTestF16_U32, AnnCagraIndexMerge) { this->testCagra(); }

INSTANTIATE_TEST_CASE_P(AnnCagraTest, AnnCagraTestF16_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnCagraIndexMergeTest,
                        AnnCagraIndexMergeTestF16_U32,
                        ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::cagra
