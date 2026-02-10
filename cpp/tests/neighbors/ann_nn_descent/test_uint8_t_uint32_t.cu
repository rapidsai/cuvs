/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_nn_descent.cuh"

#include <gtest/gtest.h>

namespace cuvs::neighbors::nn_descent {

using AnnNNDescentTestUI8_U32 = AnnNNDescentTest<float, uint8_t, std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnNNDescentTestUI8_U32, AnnNNDescent)
{
  this->testNNDescent();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(
  AnnNNDescentTest,
  AnnNNDescentTestUI8_U32,
  ::testing::ValuesIn(
    inputs));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
}  // namespace   cuvs::neighbors::nn_descent
