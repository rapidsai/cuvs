/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../dynamic_batching.cuh"

#include <cuvs/neighbors/brute_force.hpp>

namespace cuvs::neighbors::dynamic_batching {

using brute_force_float32 = dynamic_batching_test<float,
                                                  int64_t,
                                                  brute_force::index<float, float>,
                                                  brute_force::build,
                                                  brute_force::search>;

TEST_P(brute_force_float32,
       defaults)  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
{
  build_all();
  search_all();
  check_neighbors();
}

INSTANTIATE_TEST_CASE_P(dynamic_batching,
                        brute_force_float32,
                        ::testing::ValuesIn(inputs));  // NOLINT(readability-identifier-naming)

}  // namespace cuvs::neighbors::dynamic_batching
