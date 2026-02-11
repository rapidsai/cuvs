/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_nn_descent.cuh"

#include <gtest/gtest.h>

namespace cuvs::neighbors::nn_descent {

using AnnNNDescentTestI8_U32 = AnnNNDescentTest<float, int8_t, std::uint32_t>;
TEST_P(AnnNNDescentTestI8_U32,
       AnnNNDescent)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testNNDescent();
}

INSTANTIATE_TEST_CASE_P(AnnNNDescentTest, AnnNNDescentTestI8_U32, ::testing::ValuesIn(inputs));

}  // namespace   cuvs::neighbors::nn_descent
