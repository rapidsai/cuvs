/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_hnsw_ace.cuh"

namespace cuvs::neighbors::hnsw {

typedef AnnHnswAceTest<float, float, uint32_t>
  AnnHnswAceTest_float;  // NOLINT(modernize-use-using,readability-identifier-naming)
TEST_P(AnnHnswAceTest_float, AnnHnswAceBuild)
{
  this->testHnswAceBuild();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(
  AnnHnswAceTest,
  AnnHnswAceTest_float,
  ::testing::ValuesIn(
    hnsw_ace_inputs));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

// Test for memory limit fallback to disk mode
typedef AnnHnswAceTest<float, float, uint32_t>
  AnnHnswAceMemoryFallbackTest_float;  // NOLINT(modernize-use-using,readability-identifier-naming)
TEST_P(
  AnnHnswAceMemoryFallbackTest_float,
  AnnHnswAceMemoryLimitFallback)  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
{
  this->testHnswAceMemoryLimitFallback();
}

INSTANTIATE_TEST_CASE_P(
  AnnHnswAceMemoryFallbackTest,  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
  AnnHnswAceMemoryFallbackTest_float,
  ::testing::ValuesIn(hnsw_ace_memory_fallback_inputs));

}  // namespace cuvs::neighbors::hnsw
