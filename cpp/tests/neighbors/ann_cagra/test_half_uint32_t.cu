/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

using AnnCagraTestF16_U32 = AnnCagraTest<float, half, std::uint32_t>;
TEST_P(AnnCagraTestF16_U32,
       AnnCagra_U32)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra<uint32_t>();
}
TEST_P(AnnCagraTestF16_U32,
       AnnCagra_I64)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra<int64_t>();
}

using AnnCagraAddNodesTestF16_U32 = AnnCagraAddNodesTest<float, half, std::uint32_t>;
TEST_P(AnnCagraAddNodesTestF16_U32,
       AnnCagraAddNodes)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra();
}

using AnnCagraIndexMergeTestF16_U32 = AnnCagraIndexMergeTest<float, half, std::uint32_t>;
TEST_P(AnnCagraIndexMergeTestF16_U32,
       AnnCagraIndexMerge_U32)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra<uint32_t>();
}
TEST_P(AnnCagraIndexMergeTestF16_U32,
       AnnCagraIndexMerge_I64)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testCagra<int64_t>();
}

INSTANTIATE_TEST_CASE_P(AnnCagraTest, AnnCagraTestF16_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnCagraAddNodesTest,
                        AnnCagraAddNodesTestF16_U32,
                        ::testing::ValuesIn(inputs_addnode));
INSTANTIATE_TEST_CASE_P(AnnCagraIndexMergeTest,
                        AnnCagraIndexMergeTestF16_U32,
                        ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::cagra
