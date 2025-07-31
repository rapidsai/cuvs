/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#pragma once

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/scann.hpp>

#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/linalg/add.cuh>
#include <raft/matrix/gather.cuh>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <thrust/sequence.h>

namespace cuvs::neighbors::experimental::scann {

struct scann_inputs {
  uint32_t num_db_vecs = 4096;
  uint32_t dim         = 64;

  cuvs::neighbors::experimental::scann::index_params index_params;

  scann_inputs()
  {
    index_params.n_leaves            = max(32u, min(1024u, num_db_vecs / 128u));
    index_params.kmeans_n_rows_train = num_db_vecs;
    index_params.pq_n_rows_train     = num_db_vecs;
  }
};

template <typename DataT, typename IdxT>
class scann_test : public ::testing::TestWithParam<scann_inputs> {
 public:
  scann_test()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<scann_inputs>::GetParam()),
      database(0, stream_)
  {
  }

  void gen_data()
  {
    database.resize(size_t{ps.num_db_vecs} * size_t{ps.dim}, stream_);

    raft::random::RngState r(1234ULL);

    if constexpr (std::is_same<DataT, float>{}) {
      raft::random::uniform(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(1), DataT(20));
    }

    raft::resource::sync_stream(handle_);
  }

  auto build_only()
  {
    auto ipams = ps.index_params;

    auto db_view =
      raft::make_device_matrix_view<const DataT, int64_t>(database.data(), ps.num_db_vecs, ps.dim);

    return cuvs::neighbors::experimental::scann::build(handle_, ipams, db_view);
  }

  auto build_only_host_input()
  {
    auto ipams = ps.index_params;

    auto h_database = raft::make_host_matrix<DataT, int64_t>(ps.num_db_vecs, ps.dim);

    raft::copy(h_database.data_handle(), database.data(), ps.num_db_vecs * ps.dim, stream_);

    auto db_view = raft::make_host_matrix_view<const DataT, int64_t>(
      h_database.data_handle(), ps.num_db_vecs, ps.dim);

    return cuvs::neighbors::experimental::scann::build(handle_, ipams, db_view);
  }

  auto build_only_host_input_overlap()
  {
    auto ipams = ps.index_params;

    // additional stream for overlapping HtoD copy
    size_t n_streams = 2;
    raft::resource::set_cuda_stream_pool(handle_,
                                         std::make_shared<rmm::cuda_stream_pool>(n_streams));

    auto h_database = raft::make_host_matrix<DataT, int64_t>(ps.num_db_vecs, ps.dim);

    raft::copy(h_database.data_handle(), database.data(), ps.num_db_vecs * ps.dim, stream_);

    auto db_view = raft::make_host_matrix_view<const DataT, int64_t>(
      h_database.data_handle(), ps.num_db_vecs, ps.dim);

    return cuvs::neighbors::experimental::scann::build(handle_, ipams, db_view);
  }

  template <typename BuildIndex>
  void run(BuildIndex build_index)
  {
    index<DataT, IdxT> index = build_index();

    // Simple checking of dimensions of index artifacts
    auto num_subspaces   = ps.dim / ps.index_params.pq_dim;
    auto num_pq_clusters = 1 << ps.index_params.pq_bits;

    ASSERT_EQ(index.quantized_residuals().extent(0), ps.num_db_vecs);
    ASSERT_EQ(index.quantized_residuals().extent(1), num_subspaces);

    ASSERT_EQ(index.quantized_soar_residuals().extent(0), ps.num_db_vecs);
    ASSERT_EQ(index.quantized_soar_residuals().extent(1), num_subspaces);

    ASSERT_EQ(index.pq_codebook().extent(0), num_pq_clusters);
    ASSERT_EQ(index.pq_codebook().extent(1), ps.dim);

    IdxT expected_bf16_size = ps.index_params.bf16_enabled ? ps.dim * ps.num_db_vecs : 0;

    ASSERT_EQ(index.bf16_dataset().size(), expected_bf16_size);
  }

  void SetUp() override  // NOLINT
  {
    gen_data();
  }

  void TearDown() override  // NOLINT
  {
    cudaGetLastError();
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  scann_inputs ps;                      // NOLINT
  rmm::device_uvector<DataT> database;  // NOLINT
};

/* Test cases */
using test_cases_t = std::vector<scann_inputs>;

// concatenate parameter sets for different type
template <typename T>
auto operator+(const std::vector<T>& a, const std::vector<T>& b) -> std::vector<T>
{
  std::vector<T> res = a;
  res.insert(res.end(), b.begin(), b.end());
  return res;
}

template <typename B, typename A, typename F>
auto map(const std::vector<A>& xs, F f) -> std::vector<B>
{
  std::vector<B> ys(xs.size());
  std::transform(xs.begin(), xs.end(), ys.begin(), f);
  return ys;
}

inline auto with_dims(const std::vector<uint32_t>& dims) -> test_cases_t
{
  return map<scann_inputs>(dims, [](uint32_t d) {
    scann_inputs x;
    x.dim = d;
    return x;
  });
}

inline auto defaults() -> test_cases_t { return {scann_inputs{}}; }

inline auto small_dims_all_pq_bits() -> test_cases_t
{
  auto four_bit_ts = with_dims({2, 4, 6, 8, 10, 12, 14, 16, 18});

  for (auto& ts : four_bit_ts) {
    ts.index_params.pq_dim  = 2;
    ts.index_params.pq_bits = 4;
  }

  auto eight_bit_ts = with_dims({2, 4, 6, 8, 10, 12, 14, 16, 18});

  for (auto& ts : eight_bit_ts) {
    ts.index_params.pq_dim  = 2;
    ts.index_params.pq_bits = 8;
  }

  return four_bit_ts + eight_bit_ts;
}

inline auto big_dims_all_pq_bits() -> test_cases_t
{
  auto four_bit_ts = with_dims({64, 128, 256, 512, 1024, 2048});

  for (auto& ts : four_bit_ts) {
    ts.index_params.pq_dim  = 8;
    ts.index_params.pq_bits = 4;
  }

  auto eight_bit_ts = with_dims({64, 128, 256, 512, 1024, 2048});

  for (auto& ts : eight_bit_ts) {
    ts.index_params.pq_dim  = 8;
    ts.index_params.pq_bits = 8;
  }

  return four_bit_ts + eight_bit_ts;
}

inline auto bf16() -> test_cases_t
{
  scann_inputs ts;
  ts.index_params.bf16_enabled = true;

  return {ts};
}

inline auto avq() -> test_cases_t
{
  scann_inputs ts;
  ts.index_params.partitioning_eta = 2;

  return {ts};
}

inline auto soar() -> test_cases_t
{
  scann_inputs ts;
  ts.index_params.soar_lambda = 1.5;

  return {ts};
}

/* Test instantiations */

#define TEST_BUILD(type)                                \
  TEST_P(type, build) /* NOLINT */                      \
  {                                                     \
    this->run([this]() { return this->build_only(); }); \
  }

#define TEST_BUILD_HOST_INPUT(type)                                \
  TEST_P(type, build_host_input) /* NOLINT */                      \
  {                                                                \
    this->run([this]() { return this->build_only_host_input(); }); \
  }

#define TEST_BUILD_HOST_INPUT_OVERLAP(type)                                \
  TEST_P(type, build_host_input_overlap) /* NOLINT */                      \
  {                                                                        \
    this->run([this]() { return this->build_only_host_input_overlap(); }); \
  }

#define INSTANTIATE(type, vals) \
  INSTANTIATE_TEST_SUITE_P(ScaNN, type, ::testing::ValuesIn(vals)); /* NOLINT */

}  // namespace cuvs::neighbors::experimental::scann
