/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuvs/neighbors/cagra.hpp>
#include <gtest/gtest.h>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

// This test targets public API exposure and basic invariants only (shapes, in-range indices).
// Detailed optimization correctness is exercised by CAGRA build tests.

namespace {

using IdxT = uint32_t;

// Helper to create a simple synthetic KNN graph (ring-like neighbors)
auto make_ring_knn_host(int64_t num_rows, int64_t kin)
{
  auto knn_graph = raft::make_host_matrix<IdxT, int64_t>(num_rows, kin);
  for (int64_t i = 0; i < num_rows; ++i) {
    for (int64_t j = 0; j < kin; ++j) {
      knn_graph(i, j) = static_cast<IdxT>((i + j + 1) % num_rows);
    }
  }
  return knn_graph;
}

TEST(CagraOptimize, HostToHostOptimizesGraph)
{
  raft::resources res;

  constexpr int64_t num_rows = 8;
  constexpr int64_t kin      = 8;
  constexpr int64_t kout     = 4;

  auto knn_graph       = make_ring_knn_host(num_rows, kin);
  auto optimized_graph = raft::make_host_matrix<IdxT, int64_t>(num_rows, kout);

  // Test the optimize API
  cuvs::neighbors::cagra::helpers::optimize(res, knn_graph.view(), optimized_graph.view());

  // Check basic invariants
  ASSERT_EQ(optimized_graph.extent(0), num_rows);
  ASSERT_EQ(optimized_graph.extent(1), kout);

  // Check that all neighbors are valid indices
  for (int64_t i = 0; i < num_rows; ++i) {
    for (int64_t j = 0; j < kout; ++j) {
      EXPECT_LT(optimized_graph(i, j), static_cast<IdxT>(num_rows));
    }
  }
}

}  // namespace
