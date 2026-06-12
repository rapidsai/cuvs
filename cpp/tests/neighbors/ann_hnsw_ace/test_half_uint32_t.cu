/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_hnsw_ace.cuh"

namespace cuvs::neighbors::hnsw {

typedef AnnHnswAceTest<float, half, uint32_t> AnnHnswAceTest_half;
TEST_P(AnnHnswAceTest_half, AnnHnswAceBuild) { this->testHnswAceBuild(); }

INSTANTIATE_TEST_CASE_P(AnnHnswAceTest, AnnHnswAceTest_half, ::testing::ValuesIn(hnsw_ace_inputs));

typedef AnnHnswAceTest<float, half, uint32_t> AnnHnswAceMemoryFallbackTest_half;
TEST_P(AnnHnswAceMemoryFallbackTest_half, AnnHnswAceMemoryLimitFallback)
{
  this->testHnswAceMemoryLimitFallback();
}

INSTANTIATE_TEST_CASE_P(AnnHnswAceMemoryFallbackTest,
                        AnnHnswAceMemoryFallbackTest_half,
                        ::testing::ValuesIn(hnsw_ace_memory_fallback_inputs));

typedef AnnHnswAceTest<float, half, uint32_t> AnnHnswAceLayeredTest_half;
TEST_P(AnnHnswAceLayeredTest_half, AnnHnswAceLayeredBuildDeserializeSearch)
{
  this->testHnswAceLayeredBuildDeserializeSearch();
}

INSTANTIATE_TEST_CASE_P(AnnHnswAceLayeredTest,
                        AnnHnswAceLayeredTest_half,
                        ::testing::ValuesIn(hnsw_ace_layered_inputs));

}  // namespace cuvs::neighbors::hnsw
