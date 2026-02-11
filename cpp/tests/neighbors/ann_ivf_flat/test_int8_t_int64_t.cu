/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_ivf_flat.cuh"

namespace cuvs::neighbors::ivf_flat {

using AnnIVFFlatTestF_int8 = AnnIVFFlatTest<float, int8_t, int64_t>;
TEST_P(AnnIVFFlatTestF_int8,  // NOLINT(google-readability-avoid-underscore-in-googletest-name)
       AnnIVFFlat)
{
  this->testIVFFlat();
  this->testPacker();
  this->testFilter();
}

INSTANTIATE_TEST_CASE_P(AnnIVFFlatTest, AnnIVFFlatTestF_int8, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::ivf_flat
