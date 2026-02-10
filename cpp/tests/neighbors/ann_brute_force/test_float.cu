/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_brute_force.cuh"

#include <gtest/gtest.h>

namespace cuvs::neighbors::brute_force {

using AnnBruteForceTest_float =
  AnnBruteForceTest<float, float, std::int64_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnBruteForceTest_float, AnnBruteForce)
{
  this->testBruteForce();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(
  AnnBruteForceTest,
  AnnBruteForceTest_float,
  ::testing::ValuesIn(
    inputs));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

}  // namespace cuvs::neighbors::brute_force
