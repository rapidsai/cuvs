/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../all_neighbors.cuh"

namespace cuvs::neighbors::all_neighbors {

using AllNeighborsTestF = AllNeighborsTest<float, float>;  // NOLINT(readability-identifier-naming)
TEST_P(AllNeighborsTestF, AllNeighbors)
{
  this->run();
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(
  AllNeighborsSingleTest,  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
  AllNeighborsTestF,
  ::testing::ValuesIn(inputsSingle));

INSTANTIATE_TEST_CASE_P(
  AllNeighborsBatchTest,
  AllNeighborsTestF,
  ::testing::ValuesIn(
    inputsBatch));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(
  AllNeighborsSingleMutualTest,  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
  AllNeighborsTestF,
  ::testing::ValuesIn(mutualReachSingle));

INSTANTIATE_TEST_CASE_P(
  AllNeighborsBatchMutualTest,  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
  AllNeighborsTestF,
  ::testing::ValuesIn(mutualReachBatch));
}  // namespace cuvs::neighbors::all_neighbors
