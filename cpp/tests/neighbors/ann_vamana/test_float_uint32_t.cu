/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_vamana.cuh"

namespace cuvs::neighbors::vamana {

using AnnVamanaTestF_U32 =
  AnnVamanaTest<float, float, std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnVamanaTestF_U32,
       AnnVamana)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testVamana();
}  // NOLINT(readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(AnnVamanaTest,
                        AnnVamanaTestF_U32,
                        ::testing::ValuesIn(inputs));  // NOLINT(readability-identifier-naming)

}  // namespace cuvs::neighbors::vamana
