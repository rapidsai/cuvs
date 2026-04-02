/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_ivf_sq.cuh"

namespace cuvs::neighbors::ivf_sq {

typedef AnnIVFSQTest<float, float, int64_t> AnnIVFSQTestF_float;
TEST_P(AnnIVFSQTestF_float, AnnIVFSQ)
{
  this->testIVFSQ();
  this->testFilter();
}

INSTANTIATE_TEST_CASE_P(AnnIVFSQTest, AnnIVFSQTestF_float, ::testing::ValuesIn(inputs));

typedef AnnIVFSQTest<float, half, int64_t> AnnIVFSQTestF_half;
TEST_P(AnnIVFSQTestF_half, AnnIVFSQ)
{
  this->testIVFSQ();
  this->testFilter();
}

INSTANTIATE_TEST_CASE_P(AnnIVFSQTest, AnnIVFSQTestF_half, ::testing::ValuesIn(inputs_half));

}  // namespace cuvs::neighbors::ivf_sq
