/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_nn_descent.cuh"

#include <gtest/gtest.h>

namespace cuvs::neighbors::nn_descent {

typedef AnnNNDescentTest<float, float, std::uint32_t> AnnNNDescentTestF_U32;
TEST_P(AnnNNDescentTestF_U32, AnnNNDescent) { this->testNNDescent(); }

typedef AnnNNDescentDistEpiTest<float, float, std::uint32_t> AnnNNDescentTestDistEpiF_U32;
TEST_P(AnnNNDescentTestDistEpiF_U32, AnnNNDescentDistEpi) { this->testNNDescent(); }

INSTANTIATE_TEST_CASE_P(AnnNNDescentTest, AnnNNDescentTestF_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnNNDescentDistEpi,
                        AnnNNDescentTestDistEpiF_U32,
                        ::testing::ValuesIn(inputsDistEpilogue));

}  // namespace   cuvs::neighbors::nn_descent
