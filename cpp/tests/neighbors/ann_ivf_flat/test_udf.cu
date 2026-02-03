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

// ============================================================================
// Test data traits for different types
// ============================================================================

template <typename T>
struct TestDataTraits;

template <>
struct TestDataTraits<float> {
  static constexpr int64_t dim         = 4;
  static constexpr int64_t num_db_vecs = 8;

  static std::vector<float> database()
  {
    // 4-dimensional float dataset
    // Vectors arranged for easy distance verification:
    //   db[0] = [0, 0, 0, 0]  - origin
    //   db[1] = [1, 0, 0, 0]  - unit along x
    //   db[2] = [0, 1, 0, 0]  - unit along y
    //   db[3] = [0, 0, 1, 0]  - unit along z
    //   db[4] = [1, 1, 0, 0]  - diagonal in xy
    //   db[5] = [2, 0, 0, 0]  - 2 units along x
    //   db[6] = [1, 1, 1, 1]  - all ones
    //   db[7] = [3, 4, 0, 0]  - for 3-4-5 triangle
    return {
      0.0f, 0.0f, 0.0f, 0.0f,  // db[0]: origin
      1.0f, 0.0f, 0.0f, 0.0f,  // db[1]: L2 dist from origin = 1
      0.0f, 1.0f, 0.0f, 0.0f,  // db[2]: L2 dist from origin = 1
      0.0f, 0.0f, 1.0f, 0.0f,  // db[3]: L2 dist from origin = 1
      1.0f, 1.0f, 0.0f, 0.0f,  // db[4]: L2 dist from origin = 2
      2.0f, 0.0f, 0.0f, 0.0f,  // db[5]: L2 dist from origin = 4
      1.0f, 1.0f, 1.0f, 1.0f,  // db[6]: L2 dist from origin = 4
      3.0f, 4.0f, 0.0f, 0.0f,  // db[7]: L2 dist from origin = 25
    };
  }

  static std::vector<float> queries()
  {
    // query[0] = origin - nearest is db[0] (dist=0)
    // query[1] = [1,0,0,0] - nearest is db[1] (dist=0)
    return {
      0.0f,
      0.0f,
      0.0f,
      0.0f,  // query[0]: origin
      1.0f,
      0.0f,
      0.0f,
      0.0f,  // query[1]: same as db[1]
    };
  }

  // Expected: query[0] nearest is db[0] with distance 0
  static int64_t expected_nearest_idx_q0() { return 0; }
  static float expected_nearest_dist_q0() { return 0.0f; }

  // Expected: query[1] nearest is db[1] with distance 0
  static int64_t expected_nearest_idx_q1() { return 1; }
  static float expected_nearest_dist_q1() { return 0.0f; }
};

template <>
struct TestDataTraits<int8_t> {
  static constexpr int64_t dim         = 16;
  static constexpr int64_t num_db_vecs = 8;

  static std::vector<int8_t> database()
  {
    // 16-dimensional int8 dataset to test vectorized SIMD intrinsics
    return {
      // db[0]: all zeros
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      // db[1]: unit in first dim
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      // db[2]: unit in second dim
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      // db[3]: all ones - L2 dist from zeros = 16
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      // db[4]: first 12 dims are 2 - L2 dist from zeros = 48
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      0,
      0,
      0,
      0,
      // db[5]: all twos - L2 dist from zeros = 64
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      // db[6]: alternating 1,0 - L2 dist from zeros = 8
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      // db[7]: alternating 0,1 - L2 dist from zeros = 8
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
    };
  }

  static std::vector<int8_t> queries()
  {
    // query[0] = all zeros - nearest is db[0] (dist=0)
    // query[1] = all ones - nearest is db[3] (dist=0)
    return {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // query[0]
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // query[1]
    };
  }

  // Expected: query[0] nearest is db[0] with distance 0
  static int64_t expected_nearest_idx_q0() { return 0; }
  static float expected_nearest_dist_q0() { return 0.0f; }

  // Expected: query[1] nearest is db[3] with distance 0
  static int64_t expected_nearest_idx_q1() { return 3; }
  static float expected_nearest_dist_q1() { return 0.0f; }
};

// ============================================================================
// Templated test fixture
// ============================================================================

template <typename T>
class IvfFlatUdfTest : public ::testing::Test {
 protected:
  using Traits = TestDataTraits<T>;

  void SetUp() override
  {
    database_    = Traits::database();
    queries_     = Traits::queries();
    num_db_vecs_ = Traits::num_db_vecs;
    num_queries_ = 2;
    dim_         = Traits::dim;
    k_           = 4;
    n_lists_     = 2;
    n_probes_    = 2;
  }

  raft::resources handle_;
  std::vector<T> database_;
  std::vector<T> queries_;
  int64_t num_db_vecs_;
  int64_t num_queries_;
  int64_t dim_;
  int64_t k_;
  uint32_t n_lists_;
  uint32_t n_probes_;
};

using TestTypes = ::testing::Types<float, int8_t>;
TYPED_TEST_SUITE(IvfFlatUdfTest, TestTypes);

// ============================================================================
// Test: UDF L2 metric matches built-in L2 and produces correct distances
// ============================================================================

TYPED_TEST(IvfFlatUdfTest, CustomL2MatchesBuiltIn)
{
  using T      = TypeParam;
  using Traits = TestDataTraits<T>;

  auto stream = raft::resource::get_cuda_stream(this->handle_);

  // Copy data to device
  rmm::device_uvector<T> d_database(this->num_db_vecs_ * this->dim_, stream);
  rmm::device_uvector<T> d_queries(this->num_queries_ * this->dim_, stream);
  raft::copy(d_database.data(), this->database_.data(), this->database_.size(), stream);
  raft::copy(d_queries.data(), this->queries_.data(), this->queries_.size(), stream);

  auto database_view = raft::make_device_matrix_view<const T, int64_t>(
    d_database.data(), this->num_db_vecs_, this->dim_);
  auto queries_view = raft::make_device_matrix_view<const T, int64_t>(
    d_queries.data(), this->num_queries_, this->dim_);

  // Build index with L2 metric
  ivf_flat::index_params index_params;
  index_params.n_lists = this->n_lists_;
  index_params.metric  = cuvs::distance::DistanceType::L2Expanded;

  auto idx = ivf_flat::build(this->handle_, index_params, database_view);

  // Allocate output buffers
  rmm::device_uvector<int64_t> d_indices_builtin(this->num_queries_ * this->k_, stream);
  rmm::device_uvector<float> d_distances_builtin(this->num_queries_ * this->k_, stream);
  rmm::device_uvector<int64_t> d_indices_udf(this->num_queries_ * this->k_, stream);
  rmm::device_uvector<float> d_distances_udf(this->num_queries_ * this->k_, stream);

  auto indices_builtin_view = raft::make_device_matrix_view<int64_t, int64_t>(
    d_indices_builtin.data(), this->num_queries_, this->k_);
  auto distances_builtin_view = raft::make_device_matrix_view<float, int64_t>(
    d_distances_builtin.data(), this->num_queries_, this->k_);
  auto indices_udf_view = raft::make_device_matrix_view<int64_t, int64_t>(
    d_indices_udf.data(), this->num_queries_, this->k_);
  auto distances_udf_view = raft::make_device_matrix_view<float, int64_t>(
    d_distances_udf.data(), this->num_queries_, this->k_);

  // Search with built-in metric
  ivf_flat::search_params search_params_builtin;
  search_params_builtin.n_probes = this->n_probes_;

  ivf_flat::search(this->handle_,
                   search_params_builtin,
                   idx,
                   queries_view,
                   indices_builtin_view,
                   distances_builtin_view);

  // Search with custom UDF metric
  ivf_flat::search_params search_params_udf;
  search_params_udf.n_probes   = this->n_probes_;
  search_params_udf.metric_udf = custom_l2_udf();

  ivf_flat::search(
    this->handle_, search_params_udf, idx, queries_view, indices_udf_view, distances_udf_view);

  // Copy results to host
  std::vector<int64_t> h_indices_builtin(this->num_queries_ * this->k_);
  std::vector<float> h_distances_builtin(this->num_queries_ * this->k_);
  std::vector<int64_t> h_indices_udf(this->num_queries_ * this->k_);
  std::vector<float> h_distances_udf(this->num_queries_ * this->k_);

  raft::copy(
    h_indices_builtin.data(), d_indices_builtin.data(), this->num_queries_ * this->k_, stream);
  raft::copy(
    h_distances_builtin.data(), d_distances_builtin.data(), this->num_queries_ * this->k_, stream);
  raft::copy(h_indices_udf.data(), d_indices_udf.data(), this->num_queries_ * this->k_, stream);
  raft::copy(h_distances_udf.data(), d_distances_udf.data(), this->num_queries_ * this->k_, stream);
  raft::resource::sync_stream(this->handle_);

  // Verify UDF results match built-in results
  for (int64_t i = 0; i < this->num_queries_ * this->k_; ++i) {
    EXPECT_EQ(h_indices_udf[i], h_indices_builtin[i]) << "Index mismatch at position " << i;
    EXPECT_NEAR(h_distances_udf[i], h_distances_builtin[i], 1e-5f)
      << "Distance mismatch at position " << i;
  }

  // Verify expected distances for query[0]
  EXPECT_EQ(h_indices_udf[0], Traits::expected_nearest_idx_q0())
    << "Query[0] nearest neighbor index mismatch";
  EXPECT_NEAR(h_distances_udf[0], Traits::expected_nearest_dist_q0(), 1e-5f)
    << "Query[0] nearest neighbor distance mismatch";

  // Verify expected distances for query[1]
  int64_t q1_offset = this->k_;
  EXPECT_EQ(h_indices_udf[q1_offset], Traits::expected_nearest_idx_q1())
    << "Query[1] nearest neighbor index mismatch";
  EXPECT_NEAR(h_distances_udf[q1_offset], Traits::expected_nearest_dist_q1(), 1e-5f)
    << "Query[1] nearest neighbor distance mismatch";
}

}  // namespace cuvs::neighbors::ivf_flat
