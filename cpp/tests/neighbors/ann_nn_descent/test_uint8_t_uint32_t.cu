/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_nn_descent.cuh"

#include <gtest/gtest.h>

namespace cuvs::neighbors::nn_descent {

typedef AnnNNDescentTest<float, uint8_t, std::uint32_t> AnnNNDescentTestUI8_U32;
TEST_P(AnnNNDescentTestUI8_U32, AnnNNDescent) { this->testNNDescent(); }

INSTANTIATE_TEST_CASE_P(AnnNNDescentTest, AnnNNDescentTestUI8_U32, ::testing::ValuesIn(inputs));
}  // namespace   cuvs::neighbors::nn_descent
