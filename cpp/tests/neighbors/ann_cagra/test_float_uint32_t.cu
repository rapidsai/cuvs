/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

using AnnCagraTestF_U32 =
  AnnCagraTest<float, float, std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnCagraTestF_U32,
       AnnCagra_U32)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra<uint32_t>();
}  // NOLINT(readability-identifier-naming)
TEST_P(AnnCagraTestF_U32,
       AnnCagra_I64)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra<int64_t>();
}  // NOLINT(readability-identifier-naming)

using AnnCagraAddNodesTestF_U32 =
  AnnCagraAddNodesTest<float, float, std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnCagraAddNodesTestF_U32,
       AnnCagraAddNodes)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra();
}  // NOLINT(readability-identifier-naming)

using AnnCagraFilterTestF_U32 =
  AnnCagraFilterTest<float, float, std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnCagraFilterTestF_U32,
       AnnCagra)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra();
}  // NOLINT(readability-identifier-naming)

using AnnCagraIndexMergeTestF_U32 =
  AnnCagraIndexMergeTest<float, float, std::uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnCagraIndexMergeTestF_U32,
       AnnCagraIndexMerge_U32)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra<uint32_t>();
}  // NOLINT(readability-identifier-naming)
TEST_P(AnnCagraIndexMergeTestF_U32,
       AnnCagraIndexMerge_I64)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra<int64_t>();
}  // NOLINT(readability-identifier-naming)

typedef AnnCagraIndexFilteredMergeTest<float, float, std::uint32_t>  // NOLINT(modernize-use-using)
  AnnCagraIndexFilteredMergeTestF_U32;  // NOLINT(readability-identifier-naming)
TEST_P(
  AnnCagraIndexFilteredMergeTestF_U32,  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
  AnnCagraIndexFilteredMerge_U32)  // NOLINT(readability-identifier-naming,google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra<uint32_t>();
}

INSTANTIATE_TEST_CASE_P(AnnCagraTest,
                        AnnCagraTestF_U32,
                        ::testing::ValuesIn(inputs));  // NOLINT(readability-identifier-naming)
INSTANTIATE_TEST_CASE_P(AnnCagraAddNodesTest,          // NOLINT(readability-identifier-naming)
                        AnnCagraAddNodesTestF_U32,
                        ::testing::ValuesIn(inputs_addnode));
INSTANTIATE_TEST_CASE_P(AnnCagraFilterTest,  // NOLINT(readability-identifier-naming)
                        AnnCagraFilterTestF_U32,
                        ::testing::ValuesIn(inputs_filtering));
INSTANTIATE_TEST_CASE_P(AnnCagraIndexMergeTest,  // NOLINT(readability-identifier-naming)
                        AnnCagraIndexMergeTestF_U32,
                        ::testing::ValuesIn(inputs));

INSTANTIATE_TEST_CASE_P(AnnCagraIndexFilteredMergeTest,  // NOLINT(readability-identifier-naming)
                        AnnCagraIndexFilteredMergeTestF_U32,
                        ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::cagra
