/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

using AnnCagraTestU8_U32 = AnnCagraTest<float, std::uint8_t, std::uint32_t>;
TEST_P(AnnCagraTestU8_U32,
       AnnCagra)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra();
}
using AnnCagraAddNodesTestU8_U32 = AnnCagraAddNodesTest<float, std::uint8_t, std::uint32_t>;
TEST_P(AnnCagraAddNodesTestU8_U32,
       AnnCagra)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra();
}
using AnnCagraFilterTestU8_U32 = AnnCagraFilterTest<float, std::uint8_t, std::uint32_t>;
TEST_P(AnnCagraFilterTestU8_U32,
       AnnCagra)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra();
}
using AnnCagraIndexMergeTestU8_U32 = AnnCagraIndexMergeTest<float, std::uint8_t, std::uint32_t>;
TEST_P(AnnCagraIndexMergeTestU8_U32,
       AnnCagra)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra();
}

INSTANTIATE_TEST_CASE_P(AnnCagraTest, AnnCagraTestU8_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnCagraAddNodesTest,
                        AnnCagraAddNodesTestU8_U32,
                        ::testing::ValuesIn(inputs_addnode));
INSTANTIATE_TEST_CASE_P(AnnCagraFilterTest,
                        AnnCagraFilterTestU8_U32,
                        ::testing::ValuesIn(inputs_filtering));
INSTANTIATE_TEST_CASE_P(AnnCagraIndexMergeTest,
                        AnnCagraIndexMergeTestU8_U32,
                        ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::cagra
