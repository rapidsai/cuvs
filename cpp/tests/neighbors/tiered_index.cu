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

#include "../test_utils.cuh"
#include "ann_utils.cuh"

#include <cstdint>
#include <cuvs/neighbors/tiered_index.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/itertools.hpp>

#include "naive_knn.cuh"

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace cuvs::neighbors::tiered_index {

struct AnnTieredIndexInputs {
  int n_rows;
  int dim;
  cuvs::distance::DistanceType metric;
  int k;
  int n_queries;
  double min_recall;
};

inline ::std::ostream& operator<<(::std::ostream& os, const AnnTieredIndexInputs& p)
{
  os << "dataset shape=" << p.n_rows << "x" << p.dim << ", metric=" << print_metric{p.metric}
     << std::endl;
  return os;
}

template <typename UpstreamT>
class ANNTieredIndexTest : public ::testing::TestWithParam<AnnTieredIndexInputs> {
 public:
  using value_type = typename UpstreamT::value_type;
  ANNTieredIndexTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnTieredIndexInputs>::GetParam()),
      database(0, stream_),
      queries(0, stream_)
  {
  }

 protected:
  void testTieredIndex()
  {
    // Calculate the naive results
    std::vector<int64_t> indices_naive(ps.n_queries * ps.k);
    std::vector<value_type> distances_naive(ps.n_queries * ps.k);

    std::vector<value_type> queries_h(ps.n_queries * ps.dim);
    raft::update_host(queries_h.data(), queries.data(), ps.n_queries * ps.dim, stream_);

    {
      rmm::device_uvector<value_type> distances_naive_dev(ps.n_queries * ps.k, stream_);
      rmm::device_uvector<int64_t> indices_naive_dev(ps.n_queries * ps.k, stream_);
      naive_knn<value_type, value_type, int64_t>(handle_,
                                                 distances_naive_dev.data(),
                                                 indices_naive_dev.data(),
                                                 queries.data(),
                                                 database.data(),
                                                 ps.n_queries,
                                                 ps.n_rows,
                                                 ps.dim,
                                                 ps.k,
                                                 ps.metric);
      raft::update_host(
        indices_naive.data(), indices_naive_dev.data(), ps.n_queries * ps.k, stream_);
      raft::update_host(
        distances_naive.data(), distances_naive_dev.data(), ps.n_queries * ps.k, stream_);
      raft::resource::sync_stream(handle_);
    }

    // Calculate tiered index
    std::vector<int64_t> indices_tiered(ps.n_queries * ps.k);
    std::vector<value_type> distances_tiered(ps.n_queries * ps.k);
    {
      index_params<typename UpstreamT::index_params_type> build_params;
      build_params.metric                     = ps.metric;
      build_params.min_ann_rows               = 2000;
      build_params.create_ann_index_on_extend = true;

      // include 50% of the rows in the initial build - and then incrementally add
      // the rest row by row
      auto initial_rows  = ps.n_rows / 2;
      auto database_view = raft::make_device_matrix_view<const value_type, int64_t>(
        (const value_type*)database.data(), initial_rows, ps.dim);

      auto index = cuvs::neighbors::tiered_index::build(handle_, build_params, database_view);
      for (int i = initial_rows; i < ps.n_rows; ++i) {
        cuvs::neighbors::tiered_index::extend(
          handle_,
          raft::make_device_matrix_view<const value_type, int64_t>(
            (const value_type*)database.data() + i * ps.dim, 1, ps.dim),
          &index);
      }
      raft::resource::sync_stream(handle_);

      auto queries_view = raft::make_device_matrix_view<const value_type, int64_t>(
        (const value_type*)queries.data(), ps.n_queries, ps.dim);
      rmm::device_uvector<value_type> distances_tiered_dev(ps.n_queries * ps.k, stream_);
      rmm::device_uvector<int64_t> indices_tiered_dev(ps.n_queries * ps.k, stream_);

      typename UpstreamT::search_params_type search_params;

      if constexpr (std::is_same_v<ivf_flat::search_params,
                                   typename UpstreamT::search_params_type>) {
        search_params.n_probes = 128;
      }
      if constexpr (std::is_same_v<ivf_pq::search_params, typename UpstreamT::search_params_type>) {
        search_params.n_probes = 128;
      }

      auto distances_view = raft::make_device_matrix_view<value_type, int64_t>(
        (value_type*)distances_tiered_dev.data(), ps.n_queries, ps.k);
      auto indices_view = raft::make_device_matrix_view<int64_t, int64_t>(
        (int64_t*)indices_tiered_dev.data(), ps.n_queries, ps.k);

      cuvs::neighbors::tiered_index::search(
        handle_, search_params, index, queries_view, indices_view, distances_view);

      raft::update_host(
        distances_tiered.data(), distances_tiered_dev.data(), ps.n_queries * ps.k, stream_);
      raft::update_host(
        indices_tiered.data(), indices_tiered_dev.data(), ps.n_queries * ps.k, stream_);
      raft::resource::sync_stream(handle_);
    }

    double min_recall = ps.min_recall;
    EXPECT_TRUE(eval_neighbours(indices_naive,
                                indices_tiered,
                                distances_naive,
                                distances_tiered,
                                ps.n_queries,
                                ps.k,
                                0.006,
                                min_recall));
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    queries.resize(((size_t)ps.n_queries) * ps.dim, stream_);
    raft::random::RngState r(42ULL);
    raft::random::normal(
      handle_, r, database.data(), ps.n_rows * ps.dim, value_type(0.1), value_type(2.0));
    raft::random::normal(
      handle_, r, queries.data(), ps.n_queries * ps.dim, value_type(0.1), value_type(2.0));
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnTieredIndexInputs ps;
  rmm::device_uvector<value_type> database;
  rmm::device_uvector<value_type> queries;
};

const std::vector<AnnTieredIndexInputs> inputs =
  raft::util::itertools::product<AnnTieredIndexInputs>(
    {2000, 4000},   // n_rows
    {16, 29, 256},  // dim
    {cuvs::distance::DistanceType::L2Expanded,
     cuvs::distance::DistanceType::InnerProduct},  // metric
    {10},                                          // k
    {10},                                          // n_queries
    {0.85}                                         // min_recall
  );
typedef ANNTieredIndexTest<cagra::index<float, uint32_t>> CAGRA_F;
TEST_P(CAGRA_F, AnnTieredIndex) { this->testTieredIndex(); }
INSTANTIATE_TEST_CASE_P(ANNTieredIndexTest, CAGRA_F, ::testing::ValuesIn(inputs));

typedef ANNTieredIndexTest<ivf_flat::index<float, int64_t>> IvfFlat_F;
TEST_P(IvfFlat_F, AnnTieredIndex) { this->testTieredIndex(); }
INSTANTIATE_TEST_CASE_P(ANNTieredIndexTest, IvfFlat_F, ::testing::ValuesIn(inputs));

typedef ANNTieredIndexTest<ivf_pq::typed_index<float, int64_t>> IvfPq_F;
TEST_P(IvfPq_F, AnnTieredIndex) { this->testTieredIndex(); }
INSTANTIATE_TEST_CASE_P(ANNTieredIndexTest, IvfPq_F, ::testing::ValuesIn(inputs));
}  // namespace cuvs::neighbors::tiered_index
