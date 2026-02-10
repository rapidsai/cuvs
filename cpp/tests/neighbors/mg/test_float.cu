/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../mg.cuh"

namespace cuvs::neighbors::mg {

using AnnMGTestF_float = AnnMGTest<float, float>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnMGTestF_float, AnnMG)
{
  this->testAnnMG();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(
  AnnMGTest,
  AnnMGTestF_float,
  ::testing::ValuesIn(
    inputs));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

}  // namespace cuvs::neighbors::mg
