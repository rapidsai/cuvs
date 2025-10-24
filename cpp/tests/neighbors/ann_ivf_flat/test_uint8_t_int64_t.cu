/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_ivf_flat.cuh"

namespace cuvs::neighbors::ivf_flat {

typedef AnnIVFFlatTest<float, uint8_t, int64_t> AnnIVFFlatTestF_uint8;
TEST_P(AnnIVFFlatTestF_uint8, AnnIVFFlat)
{
  this->testIVFFlat();
  this->testPacker();
  this->testFilter();
}

INSTANTIATE_TEST_CASE_P(AnnIVFFlatTest, AnnIVFFlatTestF_uint8, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::ivf_flat
