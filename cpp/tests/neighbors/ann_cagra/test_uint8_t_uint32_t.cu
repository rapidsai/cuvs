/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

using AnnCagraTestU8_U32 = AnnCagraTest<float, std::uint8_t, std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnCagraTestU8_U32, AnnCagra)
{
  this->testCagra();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
using AnnCagraAddNodesTestU8_U32 = AnnCagraAddNodesTest<float, std::uint8_t, std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnCagraAddNodesTestU8_U32, AnnCagra)
{
  this->testCagra();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
using AnnCagraFilterTestU8_U32 = AnnCagraFilterTest<float, std::uint8_t, std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnCagraFilterTestU8_U32, AnnCagra)
{
  this->testCagra();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
using AnnCagraIndexMergeTestU8_U32 = AnnCagraIndexMergeTest<float, std::uint8_t, std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnCagraIndexMergeTestU8_U32, AnnCagra)
{
  this->testCagra();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(
  AnnCagraTest,
  AnnCagraTestU8_U32,
  ::testing::ValuesIn(
    inputs));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
INSTANTIATE_TEST_CASE_P(
  AnnCagraAddNodesTest,  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
  AnnCagraAddNodesTestU8_U32,
  ::testing::ValuesIn(inputs_addnode));
INSTANTIATE_TEST_CASE_P(
  AnnCagraFilterTest,  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
  AnnCagraFilterTestU8_U32,
  ::testing::ValuesIn(inputs_filtering));
INSTANTIATE_TEST_CASE_P(
  AnnCagraIndexMergeTest,  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
  AnnCagraIndexMergeTestU8_U32,
  ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::cagra
