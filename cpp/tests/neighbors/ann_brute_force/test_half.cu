/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_brute_force.cuh"

#include <gtest/gtest.h>

namespace cuvs::neighbors::brute_force {

using AnnBruteForceTest_half_float = AnnBruteForceTest<float, half, std::int64_t>;
TEST_P(AnnBruteForceTest_half_float, AnnBruteForce) { this->testBruteForce(); }

INSTANTIATE_TEST_CASE_P(AnnBruteForceTest,
                        AnnBruteForceTest_half_float,
                        ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::brute_force
