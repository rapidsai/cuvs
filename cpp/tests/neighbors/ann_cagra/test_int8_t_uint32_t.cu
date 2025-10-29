/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

typedef AnnCagraTest<float, std::int8_t, std::uint32_t> AnnCagraTestI8_U32;
TEST_P(AnnCagraTestI8_U32, AnnCagra) { this->testCagra(); }
typedef AnnCagraAddNodesTest<float, std::int8_t, std::uint32_t> AnnCagraAddNodesTestI8_U32;
TEST_P(AnnCagraAddNodesTestI8_U32, AnnCagra) { this->testCagra(); }
typedef AnnCagraFilterTest<float, std::int8_t, std::uint32_t> AnnCagraFilterTestI8_U32;
TEST_P(AnnCagraFilterTestI8_U32, AnnCagra) { this->testCagra(); }
typedef AnnCagraIndexMergeTest<float, std::int8_t, std::uint32_t> AnnCagraIndexMergeTestI8_U32;
TEST_P(AnnCagraIndexMergeTestI8_U32, AnnCagra) { this->testCagra(); }

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
