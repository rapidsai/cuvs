/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_nn_descent.cuh"

#include <gtest/gtest.h>

namespace cuvs::neighbors::nn_descent {

using AnnNNDescentTestF_U32 = AnnNNDescentTest<float, float, std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnNNDescentTestF_U32, AnnNNDescent)
{
  this->testNNDescent();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

using AnnNNDescentTestDistEpiF_U32 = AnnNNDescentDistEpiTest<float, float, std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnNNDescentTestDistEpiF_U32, AnnNNDescentDistEpi)
{
  this->testNNDescent();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(
  AnnNNDescentTest,
  AnnNNDescentTestF_U32,
  ::testing::ValuesIn(
    inputs));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
INSTANTIATE_TEST_CASE_P(
  AnnNNDescentDistEpi,  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
  AnnNNDescentTestDistEpiF_U32,
  ::testing::ValuesIn(inputsDistEpilogue));

}  // namespace   cuvs::neighbors::nn_descent
