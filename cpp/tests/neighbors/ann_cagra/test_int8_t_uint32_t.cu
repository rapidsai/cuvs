/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

using AnnCagraTestI8_U32 =
  AnnCagraTest<float, std::int8_t, std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnCagraTestI8_U32,
       AnnCagra)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra();
}  // NOLINT(readability-identifier-naming)
using AnnCagraAddNodesTestI8_U32 =
  AnnCagraAddNodesTest<float, std::int8_t, std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnCagraAddNodesTestI8_U32,
       AnnCagra)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra();
}  // NOLINT(readability-identifier-naming)
using AnnCagraFilterTestI8_U32 =
  AnnCagraFilterTest<float, std::int8_t, std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnCagraFilterTestI8_U32,
       AnnCagra)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra();
}  // NOLINT(readability-identifier-naming)
using AnnCagraIndexMergeTestI8_U32 =
  AnnCagraIndexMergeTest<float,
                         std::int8_t,
                         std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnCagraIndexMergeTestI8_U32,
       AnnCagra)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra();
}  // NOLINT(readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(AnnCagraTest,
                        AnnCagraTestI8_U32,
                        ::testing::ValuesIn(inputs));  // NOLINT(readability-identifier-naming)
INSTANTIATE_TEST_CASE_P(AnnCagraAddNodesTest,          // NOLINT(readability-identifier-naming)
                        AnnCagraAddNodesTestI8_U32,
                        ::testing::ValuesIn(inputs_addnode));
INSTANTIATE_TEST_CASE_P(AnnCagraFilterTest,  // NOLINT(readability-identifier-naming)
                        AnnCagraFilterTestI8_U32,
                        ::testing::ValuesIn(inputs_filtering));
INSTANTIATE_TEST_CASE_P(AnnCagraIndexMergeTest,  // NOLINT(readability-identifier-naming)
                        AnnCagraIndexMergeTestI8_U32,
                        ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::cagra
