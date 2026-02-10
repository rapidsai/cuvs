/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_hnsw_ace.cuh"

namespace cuvs::neighbors::hnsw {

using AnnHnswAceTest_int8_t = AnnHnswAceTest<float, int8_t, uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnHnswAceTest_int8_t, AnnHnswAceBuild)
{
  this->testHnswAceBuild();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(
  AnnHnswAceTest,  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
  AnnHnswAceTest_int8_t,
  ::testing::ValuesIn(hnsw_ace_inputs));

}  // namespace cuvs::neighbors::hnsw
