/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

typedef AnnCagraTest<float, std::uint8_t, std::uint32_t>
  AnnCagraTestU8_U32;  // NOLINT(modernize-use-using,readability-identifier-naming)
TEST_P(AnnCagraTestU8_U32, AnnCagra)
{
  this->testCagra();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
typedef AnnCagraAddNodesTest<float, std::uint8_t, std::uint32_t>
  AnnCagraAddNodesTestU8_U32;  // NOLINT(modernize-use-using,readability-identifier-naming)
TEST_P(AnnCagraAddNodesTestU8_U32, AnnCagra)
{
  this->testCagra();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
typedef AnnCagraFilterTest<float, std::uint8_t, std::uint32_t>
  AnnCagraFilterTestU8_U32;  // NOLINT(modernize-use-using,readability-identifier-naming)
TEST_P(AnnCagraFilterTestU8_U32, AnnCagra)
{
  this->testCagra();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
typedef AnnCagraIndexMergeTest<float, std::uint8_t, std::uint32_t>
  AnnCagraIndexMergeTestU8_U32;  // NOLINT(modernize-use-using,readability-identifier-naming)
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
