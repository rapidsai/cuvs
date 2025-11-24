/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

namespace cuvs::neighbors::cagra {

typedef AnnCagraTest<float, half, std::uint32_t> AnnCagraTestF16_U32;
TEST_P(AnnCagraTestF16_U32, AnnCagra_U32) { this->testCagra<uint32_t>(); }
TEST_P(AnnCagraTestF16_U32, AnnCagra_I64) { this->testCagra<int64_t>(); }

typedef AnnCagraIndexMergeTest<float, half, std::uint32_t> AnnCagraIndexMergeTestF16_U32;
TEST_P(AnnCagraIndexMergeTestF16_U32, AnnCagraIndexMerge_U32) { this->testCagra<uint32_t>(); }
TEST_P(AnnCagraIndexMergeTestF16_U32, AnnCagraIndexMerge_I64) { this->testCagra<int64_t>(); }

INSTANTIATE_TEST_CASE_P(AnnCagraTest, AnnCagraTestF16_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnCagraIndexMergeTest,
                        AnnCagraIndexMergeTestF16_U32,
                        ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::cagra
