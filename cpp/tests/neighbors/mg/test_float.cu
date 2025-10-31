/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../mg.cuh"

namespace cuvs::neighbors::mg {

typedef AnnMGTest<float, float> AnnMGTestF_float;
TEST_P(AnnMGTestF_float, AnnMG) { this->testAnnMG(); }

INSTANTIATE_TEST_CASE_P(AnnMGTest, AnnMGTestF_float, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::mg
