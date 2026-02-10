/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

using AnnCagraTestF16_U32 = AnnCagraTest<float, half, std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnCagraTestF16_U32, AnnCagra_U32)
{
  this->testCagra<uint32_t>();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
TEST_P(AnnCagraTestF16_U32, AnnCagra_I64)
{
  this->testCagra<int64_t>();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

using AnnCagraAddNodesTestF16_U32 = AnnCagraAddNodesTest<float, half, std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnCagraAddNodesTestF16_U32, AnnCagraAddNodes)
{
  this->testCagra();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

using AnnCagraIndexMergeTestF16_U32 = AnnCagraIndexMergeTest<float, half, std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnCagraIndexMergeTestF16_U32, AnnCagraIndexMerge_U32)
{
  this->testCagra<uint32_t>();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
TEST_P(AnnCagraIndexMergeTestF16_U32, AnnCagraIndexMerge_I64)
{
  this->testCagra<int64_t>();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(
  AnnCagraTest,
  AnnCagraTestF16_U32,
  ::testing::ValuesIn(
    inputs));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
INSTANTIATE_TEST_CASE_P(
  AnnCagraAddNodesTest,  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
  AnnCagraAddNodesTestF16_U32,
  ::testing::ValuesIn(inputs_addnode));
INSTANTIATE_TEST_CASE_P(
  AnnCagraIndexMergeTest,  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
  AnnCagraIndexMergeTestF16_U32,
  ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::cagra
