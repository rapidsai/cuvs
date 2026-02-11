/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_hnsw_ace.cuh"

namespace cuvs::neighbors::hnsw {

using AnnHnswAceTest_int8_t = AnnHnswAceTest<float, int8_t, uint32_t>;
TEST_P(AnnHnswAceTest_int8_t,
       AnnHnswAceBuild)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testHnswAceBuild();
}

INSTANTIATE_TEST_CASE_P(AnnHnswAceTest,
                        AnnHnswAceTest_int8_t,
                        ::testing::ValuesIn(hnsw_ace_inputs));

}  // namespace cuvs::neighbors::hnsw
