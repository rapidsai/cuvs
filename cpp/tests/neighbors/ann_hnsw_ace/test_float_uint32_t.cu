/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_hnsw_ace.cuh"

namespace cuvs::neighbors::hnsw {

typedef AnnHnswAceTest<float, float, uint32_t> AnnHnswAceTest_float;
TEST_P(AnnHnswAceTest_float, AnnHnswAceBuild) { this->testHnswAceBuild(); }

INSTANTIATE_TEST_CASE_P(AnnHnswAceTest, AnnHnswAceTest_float, ::testing::ValuesIn(hnsw_ace_inputs));

// Test with memory limits
typedef AnnHnswAceTest<float, float, uint32_t> AnnHnswAceMemoryLimitTest_float;
TEST_P(AnnHnswAceMemoryLimitTest_float, AnnHnswAceMemoryLimit) { this->testHnswAceMemoryLimit(); }

INSTANTIATE_TEST_CASE_P(AnnHnswAceMemoryLimitTest,
                        AnnHnswAceMemoryLimitTest_float,
                        ::testing::ValuesIn(hnsw_ace_memory_limit_inputs));

}  // namespace cuvs::neighbors::hnsw
