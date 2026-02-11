/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_hnsw_ace.cuh"

namespace cuvs::neighbors::hnsw {

using AnnHnswAceTest_float = AnnHnswAceTest<float, float, uint32_t>;
TEST_P(AnnHnswAceTest_float,
       AnnHnswAceBuild)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testHnswAceBuild();
}

INSTANTIATE_TEST_CASE_P(AnnHnswAceTest, AnnHnswAceTest_float, ::testing::ValuesIn(hnsw_ace_inputs));

// Test for memory limit fallback to disk mode
using AnnHnswAceMemoryFallbackTest_float = AnnHnswAceTest<float, float, uint32_t>;
TEST_P(
  AnnHnswAceMemoryFallbackTest_float,  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
  AnnHnswAceMemoryLimitFallback)
{
  this->testHnswAceMemoryLimitFallback();
}

INSTANTIATE_TEST_CASE_P(AnnHnswAceMemoryFallbackTest,
                        AnnHnswAceMemoryFallbackTest_float,
                        ::testing::ValuesIn(hnsw_ace_memory_fallback_inputs));

}  // namespace cuvs::neighbors::hnsw
