/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ann_hnsw_ace.cuh"

namespace cuvs::neighbors::hnsw {

using AnnHnswAceTest_half =
  AnnHnswAceTest<float, half, uint32_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnHnswAceTest_half,
       AnnHnswAceBuild)  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
{
  this->testHnswAceBuild();
}  // NOLINT(readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(
  AnnHnswAceTest,
  AnnHnswAceTest_half,
  ::testing::ValuesIn(hnsw_ace_inputs));  // NOLINT(readability-identifier-naming)

}  // namespace cuvs::neighbors::hnsw
