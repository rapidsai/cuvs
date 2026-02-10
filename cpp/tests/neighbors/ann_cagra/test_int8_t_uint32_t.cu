/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

typedef AnnCagraTest<float, std::int8_t, std::uint32_t>
  AnnCagraTestI8_U32;  // NOLINT(modernize-use-using,readability-identifier-naming)
TEST_P(AnnCagraTestI8_U32, AnnCagra)
{
  this->testCagra();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
typedef AnnCagraAddNodesTest<float, std::int8_t, std::uint32_t>
  AnnCagraAddNodesTestI8_U32;  // NOLINT(modernize-use-using,readability-identifier-naming)
TEST_P(AnnCagraAddNodesTestI8_U32, AnnCagra)
{
  this->testCagra();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
typedef AnnCagraFilterTest<float, std::int8_t, std::uint32_t>
  AnnCagraFilterTestI8_U32;  // NOLINT(modernize-use-using,readability-identifier-naming)
TEST_P(AnnCagraFilterTestI8_U32, AnnCagra)
{
  this->testCagra();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
typedef AnnCagraIndexMergeTest<float, std::int8_t, std::uint32_t>
  AnnCagraIndexMergeTestI8_U32;  // NOLINT(modernize-use-using,readability-identifier-naming)
TEST_P(AnnCagraIndexMergeTestI8_U32, AnnCagra)
{
  this->testCagra();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(
  AnnCagraTest,
  AnnCagraTestI8_U32,
  ::testing::ValuesIn(
    inputs));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
INSTANTIATE_TEST_CASE_P(
  AnnCagraAddNodesTest,  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
  AnnCagraAddNodesTestI8_U32,
  ::testing::ValuesIn(inputs_addnode));
INSTANTIATE_TEST_CASE_P(
  AnnCagraFilterTest,  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
  AnnCagraFilterTestI8_U32,
  ::testing::ValuesIn(inputs_filtering));
INSTANTIATE_TEST_CASE_P(
  AnnCagraIndexMergeTest,  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
  AnnCagraIndexMergeTestI8_U32,
  ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::cagra
