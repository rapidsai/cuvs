/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_vamana.cuh"

namespace cuvs::neighbors::vamana {

using AnnVamanaTestI8_U32 = AnnVamanaTest<float, int8_t, std::uint32_t>;
TEST_P(AnnVamanaTestI8_U32,
       AnnVamana)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testVamana();
}

INSTANTIATE_TEST_CASE_P(AnnVamanaTest, AnnVamanaTestI8_U32, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::vamana
