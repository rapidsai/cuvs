/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../mg.cuh"

namespace cuvs::neighbors::mg {

using AnnMGTestF_float = AnnMGTest<float, float>;
TEST_P(AnnMGTestF_float, AnnMG)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testAnnMG();
}

INSTANTIATE_TEST_CASE_P(AnnMGTest, AnnMGTestF_float, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::mg
