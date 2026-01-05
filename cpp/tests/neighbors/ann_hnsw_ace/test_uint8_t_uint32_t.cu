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

}  // namespace cuvs::neighbors::hnsw
