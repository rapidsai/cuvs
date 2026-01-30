/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <rmm/device_uvector.hpp>

#include <vector>

namespace cuvs::neighbors::ivf_flat {

// ============================================================================
// Define custom metrics using the UDF macro
// ============================================================================

// Custom L2 (squared Euclidean) metric - should match built-in L2
CUVS_METRIC(custom_l2, { acc += squared_diff(x, y); })

// Custom inner product metric - should match built-in InnerProduct
// Note: Built-in uses negative inner product (larger similarity = smaller distance)
CUVS_METRIC(custom_inner_product, { acc -= dot_product(x, y); })

// Custom L1 (Manhattan) metric
CUVS_METRIC(custom_l1, { acc += abs_diff(x, y); })

// ============================================================================
// Test fixture
// ============================================================================

class IvfFlatUdfTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Hardcoded 2D dataset for easy manual verification
    // 8 database vectors in 4 dimensions
    //
    // Vectors arranged so we can easily verify distances:
    //   db[0] = [0, 0, 0, 0]  - origin
    //   db[1] = [1, 0, 0, 0]  - unit along x
    //   db[2] = [0, 1, 0, 0]  - unit along y
    //   db[3] = [0, 0, 1, 0]  - unit along z
    //   db[4] = [1, 1, 0, 0]  - diagonal in xy
    //   db[5] = [2, 0, 0, 0]  - 2 units along x
    //   db[6] = [1, 1, 1, 1]  - all ones
    //   db[7] = [3, 4, 0, 0]  - for 3-4-5 triangle verification
    //
    database_ = {
      0.0f, 0.0f, 0.0f, 0.0f,  // db[0]: origin
      1.0f, 0.0f, 0.0f, 0.0f,  // db[1]: L2 dist from origin = 1
      0.0f, 1.0f, 0.0f, 0.0f,  // db[2]: L2 dist from origin = 1
      0.0f, 0.0f, 1.0f, 0.0f,  // db[3]: L2 dist from origin = 1
      1.0f, 1.0f, 0.0f, 0.0f,  // db[4]: L2 dist from origin = sqrt(2) ≈ 1.414
      2.0f, 0.0f, 0.0f, 0.0f,  // db[5]: L2 dist from origin = 2
      1.0f, 1.0f, 1.0f, 1.0f,  // db[6]: L2 dist from origin = 2
      3.0f, 4.0f, 0.0f, 0.0f,  // db[7]: L2 dist from origin = 5
    };

    // Query vectors
    // query[0] = origin - nearest neighbors should be db[0], then db[1,2,3] (all dist=1)
    // query[1] = [1,0,0,0] - nearest is db[1] (dist=0), then db[0,4] (dist=1)
    queries_ = {
      0.0f,
      0.0f,
      0.0f,
      0.0f,  // query[0]: origin
      1.0f,
      0.0f,
      0.0f,
      0.0f,  // query[1]: same as db[1]
    };

    num_db_vecs_ = 8;
    num_queries_ = 2;
    dim_         = 4;
    k_           = 4;
    n_lists_     = 2;  // Small number for this tiny dataset
    n_probes_    = 2;  // Search all clusters
  }

  raft::resources handle_;
  std::vector<float> database_;
  std::vector<float> queries_;
  int64_t num_db_vecs_;
  int64_t num_queries_;
  int64_t dim_;
  int64_t k_;
  uint32_t n_lists_;
  uint32_t n_probes_;
};

// ============================================================================
// Test: UDF L2 metric matches built-in L2
// ============================================================================

TEST_F(IvfFlatUdfTest, CustomL2MatchesBuiltIn)
{
  auto stream = raft::resource::get_cuda_stream(handle_);

  // Copy data to device
  rmm::device_uvector<float> d_database(num_db_vecs_ * dim_, stream);
  rmm::device_uvector<float> d_queries(num_queries_ * dim_, stream);
  raft::copy(d_database.data(), database_.data(), database_.size(), stream);
  raft::copy(d_queries.data(), queries_.data(), queries_.size(), stream);

  auto database_view =
    raft::make_device_matrix_view<const float, int64_t>(d_database.data(), num_db_vecs_, dim_);
  auto queries_view =
    raft::make_device_matrix_view<const float, int64_t>(d_queries.data(), num_queries_, dim_);

  // Build index with L2 metric
  ivf_flat::index_params index_params;
  index_params.n_lists = n_lists_;
  index_params.metric  = cuvs::distance::DistanceType::L2Expanded;

  auto idx = ivf_flat::build(handle_, index_params, database_view);

  // Allocate output buffers
  rmm::device_uvector<int64_t> d_indices_builtin(num_queries_ * k_, stream);
  rmm::device_uvector<float> d_distances_builtin(num_queries_ * k_, stream);
  rmm::device_uvector<int64_t> d_indices_udf(num_queries_ * k_, stream);
  rmm::device_uvector<float> d_distances_udf(num_queries_ * k_, stream);

  auto indices_builtin_view =
    raft::make_device_matrix_view<int64_t, int64_t>(d_indices_builtin.data(), num_queries_, k_);
  auto distances_builtin_view =
    raft::make_device_matrix_view<float, int64_t>(d_distances_builtin.data(), num_queries_, k_);
  auto indices_udf_view =
    raft::make_device_matrix_view<int64_t, int64_t>(d_indices_udf.data(), num_queries_, k_);
  auto distances_udf_view =
    raft::make_device_matrix_view<float, int64_t>(d_distances_udf.data(), num_queries_, k_);

  // Search with built-in metric
  ivf_flat::search_params search_params_builtin;
  search_params_builtin.n_probes = n_probes_;

  ivf_flat::search(handle_,
                   search_params_builtin,
                   idx,
                   queries_view,
                   indices_builtin_view,
                   distances_builtin_view);

  // Search with custom UDF metric
  ivf_flat::search_params search_params_udf;
  search_params_udf.n_probes   = n_probes_;
  search_params_udf.metric_udf = custom_l2_udf();

  ivf_flat::search(
    handle_, search_params_udf, idx, queries_view, indices_udf_view, distances_udf_view);

  // Copy results to host
  std::vector<int64_t> h_indices_builtin(num_queries_ * k_);
  std::vector<float> h_distances_builtin(num_queries_ * k_);
  std::vector<int64_t> h_indices_udf(num_queries_ * k_);
  std::vector<float> h_distances_udf(num_queries_ * k_);

  raft::copy(h_indices_builtin.data(), d_indices_builtin.data(), num_queries_ * k_, stream);
  raft::copy(h_distances_builtin.data(), d_distances_builtin.data(), num_queries_ * k_, stream);
  raft::copy(h_indices_udf.data(), d_indices_udf.data(), num_queries_ * k_, stream);
  raft::copy(h_distances_udf.data(), d_distances_udf.data(), num_queries_ * k_, stream);
  raft::resource::sync_stream(handle_);

  // Verify UDF results match built-in results
  for (int64_t i = 0; i < num_queries_ * k_; ++i) {
    EXPECT_EQ(h_indices_udf[i], h_indices_builtin[i])
      << "Index mismatch at position " << i << ": UDF=" << h_indices_udf[i]
      << ", builtin=" << h_indices_builtin[i];
    EXPECT_NEAR(h_distances_udf[i], h_distances_builtin[i], 1e-5f)
      << "Distance mismatch at position " << i << ": UDF=" << h_distances_udf[i]
      << ", builtin=" << h_distances_builtin[i];
  }

  // Additional verification: check expected distances for query[0] (origin)
  // The nearest neighbor should be db[0] (origin) with distance 0
  EXPECT_EQ(h_indices_udf[0], 0) << "Nearest to origin should be db[0]";
  EXPECT_NEAR(h_distances_udf[0], 0.0f, 1e-5f) << "Distance from origin to origin should be 0";
}

// ============================================================================
// Test: UDF produces correct L2 distances (manual verification)
// ============================================================================

TEST_F(IvfFlatUdfTest, CustomL2CorrectDistances)
{
  auto stream = raft::resource::get_cuda_stream(handle_);

  // Copy data to device
  rmm::device_uvector<float> d_database(num_db_vecs_ * dim_, stream);
  rmm::device_uvector<float> d_queries(num_queries_ * dim_, stream);
  raft::copy(d_database.data(), database_.data(), database_.size(), stream);
  raft::copy(d_queries.data(), queries_.data(), queries_.size(), stream);

  auto database_view =
    raft::make_device_matrix_view<const float, int64_t>(d_database.data(), num_db_vecs_, dim_);
  auto queries_view =
    raft::make_device_matrix_view<const float, int64_t>(d_queries.data(), num_queries_, dim_);

  // Build index
  ivf_flat::index_params index_params;
  index_params.n_lists = n_lists_;
  index_params.metric  = cuvs::distance::DistanceType::L2Expanded;

  auto idx = ivf_flat::build(handle_, index_params, database_view);

  // Allocate output
  rmm::device_uvector<int64_t> d_indices(num_queries_ * k_, stream);
  rmm::device_uvector<float> d_distances(num_queries_ * k_, stream);

  auto indices_view =
    raft::make_device_matrix_view<int64_t, int64_t>(d_indices.data(), num_queries_, k_);
  auto distances_view =
    raft::make_device_matrix_view<float, int64_t>(d_distances.data(), num_queries_, k_);

  // Search with UDF
  ivf_flat::search_params search_params;
  search_params.n_probes   = n_probes_;
  search_params.metric_udf = custom_l2_udf();

  ivf_flat::search(handle_, search_params, idx, queries_view, indices_view, distances_view);

  // Copy to host
  std::vector<int64_t> h_indices(num_queries_ * k_);
  std::vector<float> h_distances(num_queries_ * k_);
  raft::copy(h_indices.data(), d_indices.data(), num_queries_ * k_, stream);
  raft::copy(h_distances.data(), d_distances.data(), num_queries_ * k_, stream);
  raft::resource::sync_stream(handle_);

  // Verify query[1] = [1,0,0,0]
  // Expected: db[1] at distance 0 (exact match)
  //           db[0] at distance 1 (squared L2)
  //           db[4]=[1,1,0,0] at distance 1 (squared L2)
  //           db[2]=[0,1,0,0] at distance 2 (squared L2)
  int64_t q1_offset = k_;  // Results for query[1] start at index k_
  EXPECT_EQ(h_indices[q1_offset], 1) << "Query[1] nearest should be db[1] (exact match)";
  EXPECT_NEAR(h_distances[q1_offset], 0.0f, 1e-5f) << "Distance should be 0 for exact match";
}

// ============================================================================
// Test: Inner product UDF
// ============================================================================

TEST_F(IvfFlatUdfTest, CustomInnerProductMatchesBuiltIn)
{
  auto stream = raft::resource::get_cuda_stream(handle_);

  // Copy data to device
  rmm::device_uvector<float> d_database(num_db_vecs_ * dim_, stream);
  rmm::device_uvector<float> d_queries(num_queries_ * dim_, stream);
  raft::copy(d_database.data(), database_.data(), database_.size(), stream);
  raft::copy(d_queries.data(), queries_.data(), queries_.size(), stream);

  auto database_view =
    raft::make_device_matrix_view<const float, int64_t>(d_database.data(), num_db_vecs_, dim_);
  auto queries_view =
    raft::make_device_matrix_view<const float, int64_t>(d_queries.data(), num_queries_, dim_);

  // Build index with InnerProduct metric
  ivf_flat::index_params index_params;
  index_params.n_lists = n_lists_;
  index_params.metric  = cuvs::distance::DistanceType::InnerProduct;

  auto idx = ivf_flat::build(handle_, index_params, database_view);

  // Allocate output buffers
  rmm::device_uvector<int64_t> d_indices_builtin(num_queries_ * k_, stream);
  rmm::device_uvector<float> d_distances_builtin(num_queries_ * k_, stream);
  rmm::device_uvector<int64_t> d_indices_udf(num_queries_ * k_, stream);
  rmm::device_uvector<float> d_distances_udf(num_queries_ * k_, stream);

  auto indices_builtin_view =
    raft::make_device_matrix_view<int64_t, int64_t>(d_indices_builtin.data(), num_queries_, k_);
  auto distances_builtin_view =
    raft::make_device_matrix_view<float, int64_t>(d_distances_builtin.data(), num_queries_, k_);
  auto indices_udf_view =
    raft::make_device_matrix_view<int64_t, int64_t>(d_indices_udf.data(), num_queries_, k_);
  auto distances_udf_view =
    raft::make_device_matrix_view<float, int64_t>(d_distances_udf.data(), num_queries_, k_);

  // Search with built-in metric
  ivf_flat::search_params search_params_builtin;
  search_params_builtin.n_probes = n_probes_;

  ivf_flat::search(handle_,
                   search_params_builtin,
                   idx,
                   queries_view,
                   indices_builtin_view,
                   distances_builtin_view);

  // Search with custom UDF metric
  ivf_flat::search_params search_params_udf;
  search_params_udf.n_probes   = n_probes_;
  search_params_udf.metric_udf = custom_inner_product_udf();

  ivf_flat::search(
    handle_, search_params_udf, idx, queries_view, indices_udf_view, distances_udf_view);

  // Copy results to host
  std::vector<int64_t> h_indices_builtin(num_queries_ * k_);
  std::vector<float> h_distances_builtin(num_queries_ * k_);
  std::vector<int64_t> h_indices_udf(num_queries_ * k_);
  std::vector<float> h_distances_udf(num_queries_ * k_);

  raft::copy(h_indices_builtin.data(), d_indices_builtin.data(), num_queries_ * k_, stream);
  raft::copy(h_distances_builtin.data(), d_distances_builtin.data(), num_queries_ * k_, stream);
  raft::copy(h_indices_udf.data(), d_indices_udf.data(), num_queries_ * k_, stream);
  raft::copy(h_distances_udf.data(), d_distances_udf.data(), num_queries_ * k_, stream);
  raft::resource::sync_stream(handle_);

  // Verify UDF results match built-in results
  for (int64_t i = 0; i < num_queries_ * k_; ++i) {
    EXPECT_EQ(h_indices_udf[i], h_indices_builtin[i]) << "Index mismatch at position " << i;
    EXPECT_NEAR(h_distances_udf[i], h_distances_builtin[i], 1e-5f)
      << "Distance mismatch at position " << i;
  }
}

}  // namespace cuvs::neighbors::ivf_flat
