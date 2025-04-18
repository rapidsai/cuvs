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

#include "../ann_nn_descent.cuh"

#include <gtest/gtest.h>

namespace cuvs::neighbors::nn_descent {

typedef AnnNNDescentTest<float, float, std::uint32_t> AnnNNDescentTestF_U32;
TEST_P(AnnNNDescentTestF_U32, AnnNNDescent) { this->testNNDescent(); }

typedef AnnNNDescentBatchTest<float, float, std::uint32_t> AnnNNDescentBatchTestF_U32;
TEST_P(AnnNNDescentBatchTestF_U32, AnnNNDescentBatch) { this->testNNDescentBatch(); }

INSTANTIATE_TEST_CASE_P(AnnNNDescentTest, AnnNNDescentTestF_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnNNDescentBatchTest,
                        AnnNNDescentBatchTestF_U32,
                        ::testing::ValuesIn(inputsBatch));

}  // namespace   cuvs::neighbors::nn_descent
