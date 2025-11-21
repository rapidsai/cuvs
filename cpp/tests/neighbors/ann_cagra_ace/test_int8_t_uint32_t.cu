/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra_ace.cuh"

namespace cuvs::neighbors::cagra {

typedef AnnCagraAceTest<float, std::int8_t, std::uint32_t> AnnCagraAceTestI8_U32;
TEST_P(AnnCagraAceTestI8_U32, AnnCagraAce) { this->testAce(); }

INSTANTIATE_TEST_CASE_P(AnnCagraAceTest, AnnCagraAceTestI8_U32, ::testing::ValuesIn(ace_inputs));

}  // namespace cuvs::neighbors::cagra
