/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

using AnnCagraTestI8_U32 = AnnCagraTest<float, std::int8_t, std::uint32_t>;
TEST_P(AnnCagraTestI8_U32,
       AnnCagra)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra();
}
using AnnCagraAddNodesTestI8_U32 = AnnCagraAddNodesTest<float, std::int8_t, std::uint32_t>;
TEST_P(AnnCagraAddNodesTestI8_U32,
       AnnCagra)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra();
}
using AnnCagraFilterTestI8_U32 = AnnCagraFilterTest<float, std::int8_t, std::uint32_t>;
TEST_P(AnnCagraFilterTestI8_U32,
       AnnCagra)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra();
}
using AnnCagraIndexMergeTestI8_U32 = AnnCagraIndexMergeTest<float, std::int8_t, std::uint32_t>;
TEST_P(AnnCagraIndexMergeTestI8_U32,
       AnnCagra)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra();
}

INSTANTIATE_TEST_CASE_P(AnnCagraTest, AnnCagraTestI8_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnCagraAddNodesTest,
                        AnnCagraAddNodesTestI8_U32,
                        ::testing::ValuesIn(inputs_addnode));
INSTANTIATE_TEST_CASE_P(AnnCagraFilterTest,
                        AnnCagraFilterTestI8_U32,
                        ::testing::ValuesIn(inputs_filtering));
INSTANTIATE_TEST_CASE_P(AnnCagraIndexMergeTest,
                        AnnCagraIndexMergeTestI8_U32,
                        ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::cagra
