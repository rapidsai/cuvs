/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_ivf_flat.cuh"

namespace cuvs::neighbors::ivf_flat {

using AnnIVFFlatTestF_half =
  AnnIVFFlatTest<float, half, int64_t>;  // NOLINT(readability-identifier-naming)
TEST_P(AnnIVFFlatTestF_half,  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
       AnnIVFFlat)            // NOLINT(readability-identifier-naming)
{
  this->testIVFFlat();
  this->testPacker();
  this->testFilter();
}

INSTANTIATE_TEST_CASE_P(AnnIVFFlatTest,
                        AnnIVFFlatTestF_half,
                        ::testing::ValuesIn(inputs));  // NOLINT(readability-identifier-naming)

}  // namespace cuvs::neighbors::ivf_flat
