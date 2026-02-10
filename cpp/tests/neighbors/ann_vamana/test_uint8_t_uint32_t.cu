/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_vamana.cuh"

namespace cuvs::neighbors::vamana {

typedef AnnVamanaTest<float, uint8_t, std::uint32_t>
  AnnVamanaTestU8_U32;  // NOLINT(modernize-use-using,readability-identifier-naming)
TEST_P(AnnVamanaTestU8_U32, AnnVamana)
{
  this->testVamana();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(
  AnnVamanaTest,
  AnnVamanaTestU8_U32,
  ::testing::ValuesIn(
    inputs));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

}  // namespace cuvs::neighbors::vamana
