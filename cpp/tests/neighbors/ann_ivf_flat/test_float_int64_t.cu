/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_ivf_flat.cuh"

namespace cuvs::neighbors::ivf_flat {

typedef AnnIVFFlatTest<float, float, int64_t> AnnIVFFlatTestF_float;
TEST_P(AnnIVFFlatTestF_float, AnnIVFFlat)
{
  this->testIVFFlat();
  this->testPacker();
  this->testFilter();
}

INSTANTIATE_TEST_CASE_P(AnnIVFFlatTest, AnnIVFFlatTestF_float, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::ivf_flat
