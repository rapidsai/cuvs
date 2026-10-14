/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_scann.cuh"

#include <raft/matrix/init.cuh>

namespace cuvs::neighbors::scann {

using f32_i64 = scann_test<float, int64_t>;

TEST_BUILD(f32_i64)
TEST_BUILD_HOST_INPUT(f32_i64)
TEST_BUILD_HOST_INPUT_OVERLAP(f32_i64);

INSTANTIATE(f32_i64,
            defaults() + small_dims_all_pq_bits() + big_dims_all_pq_bits() + bf16() + bf16_avq() +
              avq() + soar() + two_level());

// Parameter validation: build must reject invalid index parameters. These checks
// run in the index constructor (before any training), so build throws immediately.
TEST(AnnScaNNTest, InvalidBuildParams)  // NOLINT
{
  raft::resources handle;

  constexpr int64_t rows = 256;
  constexpr int64_t dim  = 16;

  auto data = raft::make_device_matrix<float, int64_t>(handle, rows, dim);
  raft::matrix::fill(handle, data.view(), 1.0f);
  auto data_view =
    raft::make_device_matrix_view<const float, int64_t>(data.data_handle(), rows, dim);

  auto base_params = []() {
    index_params p;
    p.n_leaves            = 4;
    p.n_coarse_clusters   = 0;
    p.kmeans_n_rows_train = rows;
    p.pq_n_rows_train     = rows;
    p.pq_dim              = 2;
    p.pq_bits             = 8;
    return p;
  };

  // pq_bits must be 4 or 8
  {
    auto p    = base_params();
    p.pq_bits = 5;
    EXPECT_THROW(cuvs::neighbors::scann::build(handle, p, data_view), raft::exception);
  }
  // dataset dimension must be divisible by pq_dim
  {
    auto p   = base_params();
    p.pq_dim = 3;
    EXPECT_THROW(cuvs::neighbors::scann::build(handle, p, data_view), raft::exception);
  }
  // pq_dim must not exceed the dataset dimension
  {
    auto p   = base_params();
    p.pq_dim = 32;
    EXPECT_THROW(cuvs::neighbors::scann::build(handle, p, data_view), raft::exception);
  }
  // n_coarse_clusters must be strictly less than n_leaves
  {
    auto p              = base_params();
    p.n_coarse_clusters = 8;
    p.n_leaves          = 4;
    EXPECT_THROW(cuvs::neighbors::scann::build(handle, p, data_view), raft::exception);
  }
}

}  // namespace cuvs::neighbors::scann
