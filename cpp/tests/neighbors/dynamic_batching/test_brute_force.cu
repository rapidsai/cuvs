/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
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

TEST_P(brute_force_float32, defaults)
{
  build_all();
  search_all();
  check_neighbors();
}

INSTANTIATE_TEST_CASE_P(dynamic_batching, brute_force_float32, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::dynamic_batching
