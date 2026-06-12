/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_hnsw_ace.cuh"

namespace cuvs::neighbors::hnsw {

typedef AnnHnswAceTest<float, uint8_t, uint32_t> AnnHnswAceTest_uint8_t;
TEST_P(AnnHnswAceTest_uint8_t, AnnHnswAceBuild) { this->testHnswAceBuild(); }

INSTANTIATE_TEST_CASE_P(AnnHnswAceTest,
                        AnnHnswAceTest_uint8_t,
                        ::testing::ValuesIn(hnsw_ace_inputs));

typedef AnnHnswAceTest<float, uint8_t, uint32_t> AnnHnswAceMemoryFallbackTest_uint8_t;
TEST_P(AnnHnswAceMemoryFallbackTest_uint8_t, AnnHnswAceMemoryLimitFallback)
{
  this->testHnswAceMemoryLimitFallback();
}

INSTANTIATE_TEST_CASE_P(AnnHnswAceMemoryFallbackTest,
                        AnnHnswAceMemoryFallbackTest_uint8_t,
                        ::testing::ValuesIn(hnsw_ace_memory_fallback_inputs));

typedef AnnHnswAceTest<float, uint8_t, uint32_t> AnnHnswAceLayeredTest_uint8_t;
TEST_P(AnnHnswAceLayeredTest_uint8_t, AnnHnswAceLayeredBuildDeserializeSearch)
{
  this->testHnswAceLayeredBuildDeserializeSearch();
}

INSTANTIATE_TEST_CASE_P(AnnHnswAceLayeredTest,
                        AnnHnswAceLayeredTest_uint8_t,
                        ::testing::ValuesIn(hnsw_ace_layered_inputs));

typedef AnnHnswAceTest<float, uint8_t, uint32_t> AnnHnswAceMaterializeTest_uint8_t;
TEST_P(AnnHnswAceMaterializeTest_uint8_t, AnnHnswAceLayeredMaterializeToHnswlib)
{
  this->testHnswAceLayeredMaterializeToHnswlib();
}

INSTANTIATE_TEST_CASE_P(AnnHnswAceMaterializeTest,
                        AnnHnswAceMaterializeTest_uint8_t,
                        ::testing::ValuesIn(hnsw_ace_layered_inputs));

}  // namespace cuvs::neighbors::hnsw
