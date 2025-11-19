/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../all_neighbors.cuh"

namespace cuvs::neighbors::all_neighbors {

typedef AllNeighborsTest<float, float> AllNeighborsTestF;
TEST_P(AllNeighborsTestF, AllNeighbors) { this->run(); }

INSTANTIATE_TEST_CASE_P(AllNeighborsSingleTest,
                        AllNeighborsTestF,
                        ::testing::ValuesIn(inputsSingle));

INSTANTIATE_TEST_CASE_P(AllNeighborsBatchTest, AllNeighborsTestF, ::testing::ValuesIn(inputsBatch));

INSTANTIATE_TEST_CASE_P(AllNeighborsSingleMutualTest,
                        AllNeighborsTestF,
                        ::testing::ValuesIn(mutualReachSingle));

INSTANTIATE_TEST_CASE_P(AllNeighborsBatchMutualTest,
                        AllNeighborsTestF,
                        ::testing::ValuesIn(mutualReachBatch));
}  // namespace cuvs::neighbors::all_neighbors
