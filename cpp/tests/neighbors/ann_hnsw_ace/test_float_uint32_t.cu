/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_hnsw_ace.cuh"

namespace cuvs::neighbors::hnsw {

using AnnHnswAceTest_float =
  AnnHnswAceTest<float, float, uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnHnswAceTest_float,
       AnnHnswAceBuild)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testHnswAceBuild();
}  // NOLINT(readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(
  AnnHnswAceTest,
  AnnHnswAceTest_float,
  ::testing::ValuesIn(hnsw_ace_inputs));  // NOLINT(readability-identifier-naming)

// Test for memory limit fallback to disk mode
using AnnHnswAceMemoryFallbackTest_float =
  AnnHnswAceTest<float, float, uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(
  AnnHnswAceMemoryFallbackTest_float,  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
  AnnHnswAceMemoryLimitFallback)       // NOLINT(readability-identifier-naming)
{
  this->testHnswAceMemoryLimitFallback();
}

INSTANTIATE_TEST_CASE_P(AnnHnswAceMemoryFallbackTest,  // NOLINT(readability-identifier-naming)
                        AnnHnswAceMemoryFallbackTest_float,
                        ::testing::ValuesIn(hnsw_ace_memory_fallback_inputs));

}  // namespace cuvs::neighbors::hnsw
