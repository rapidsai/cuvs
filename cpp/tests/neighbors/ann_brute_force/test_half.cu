/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_brute_force.cuh"

#include <gtest/gtest.h>

namespace cuvs::neighbors::brute_force {

using AnnBruteForceTest_half_float = AnnBruteForceTest<float, half, std::int64_t>;
TEST_P(AnnBruteForceTest_half_float,
       AnnBruteForce)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testBruteForce();
}  // NOLINT(readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(AnnBruteForceTest,  // NOLINT(readability-identifier-naming)
                        AnnBruteForceTest_half_float,
                        ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::brute_force
