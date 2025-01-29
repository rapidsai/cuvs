/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <raft/core/resource/cuda_stream.hpp>

#include "naive_knn.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/vamana.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/linalg/add.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/itertools.hpp>

#include <rmm/device_buffer.hpp>

#include <gtest/gtest.h>

#include <thrust/sequence.h>

#include <cstddef>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace cuvs::neighbors::experimental::vamana {

struct edge_op {
  template <typename Type, typename... UnusedArgs>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in, UnusedArgs...) const
  {
    return in == raft::upper_bound<Type>() ? Type(0) : in;
  }
};

struct AnnVamanaInputs {
  int n_rows;
  int dim;
  int graph_degree;
  int visited_size;
  double max_fraction;
  cuvs::distance::DistanceType metric;
  bool host_dataset;

  // cagra search params
  int n_queries;
  int k;
  cagra::search_algo algo;
  int max_queries;
  int itopk_size;
  int search_width;
  double min_recall;
};

template <typename DataT, typename IdxT>
inline void CheckGraph(vamana::index<DataT, IdxT>* index_,
                       AnnVamanaInputs inputs,
                       cudaStream_t stream)
{
  EXPECT_TRUE(index_->graph().size() == (inputs.n_rows * inputs.graph_degree));
  EXPECT_TRUE(index_->graph().extent(0) == inputs.n_rows);
  EXPECT_TRUE(index_->graph().extent(1) == inputs.graph_degree);

  // Copy graph to host
  auto h_graph = raft::make_host_matrix<IdxT, int64_t>(inputs.n_rows, inputs.graph_degree);
  raft::copy(h_graph.data_handle(), index_->graph().data_handle(), index_->graph().size(), stream);

  size_t edge_count = 0;
  int max_degree    = 0;
  for (int i = 0; i < h_graph.extent(0); i++) {
    int temp_degree = 0;
    for (int j = 0; j < h_graph.extent(1); j++) {
      if (h_graph(i, j) < (uint32_t)(inputs.n_rows)) temp_degree++;
    }
    if (temp_degree > max_degree) max_degree = temp_degree;
    edge_count += (size_t)temp_degree;
  }

  // Tests for acceptable range of edges - low dim can also impact this
  // Minimum expected maximum degree across the whole graph
  EXPECT_TRUE(max_degree >= std::min(inputs.graph_degree, inputs.dim));

  float max_edges = (float)(inputs.n_rows * std::min(inputs.graph_degree, inputs.dim));

  RAFT_LOG_INFO("dim:%d, degree:%d, visited_size:%d, Total edges:%lu, Maximum edges:%lu",
                inputs.dim,
                inputs.graph_degree,
                inputs.visited_size,
                edge_count,
                (size_t)max_edges);

  // Graph won't always be full, but <75% is very unlikely
  EXPECT_TRUE(((float)edge_count / max_edges) > 0.75);
}

template <typename DistanceT, typename DataT, typename IdxT>
class AnnVamanaTest : public ::testing::TestWithParam<AnnVamanaInputs> {
 public:
  AnnVamanaTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnVamanaInputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

 protected:
  void testVamana()
  {
    vamana::index_params index_params;
    index_params.metric       = ps.metric;
    index_params.graph_degree = ps.graph_degree;
    index_params.visited_size = ps.visited_size;
    index_params.max_fraction = ps.max_fraction;

    auto database_view = raft::make_device_matrix_view<const DataT, int64_t>(
      (const DataT*)database.data(), ps.n_rows, ps.dim);

    vamana::index<DataT, IdxT> index(handle_);
    if (ps.host_dataset) {
      auto database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
      raft::copy(database_host.data_handle(), database.data(), database.size(), stream_);
      auto database_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
        (const DataT*)database_host.data_handle(), ps.n_rows, ps.dim);

      index = vamana::build(handle_, index_params, database_host_view);
    } else {
      index = vamana::build(handle_, index_params, database_view);
    };

    CheckGraph<DataT, IdxT>(&index, ps, stream_);

    vamana::serialize(handle_, "vamana_index", index);

    // Test recall by searching with CAGRA search
    if (ps.graph_degree < 256) {  // CAGRA search result buffer cannot support larger graph degree
      size_t queries_size = ps.n_queries * ps.k;
      std::vector<IdxT> indices_Cagra(queries_size);
      std::vector<IdxT> indices_naive(queries_size);
      std::vector<DistanceT> distances_Cagra(queries_size);
      std::vector<DistanceT> distances_naive(queries_size);

      {
        rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, stream_);
        rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);

        cuvs::neighbors::naive_knn<DistanceT, DataT, IdxT>(handle_,
                                                           distances_naive_dev.data(),
                                                           indices_naive_dev.data(),
                                                           search_queries.data(),
                                                           database.data(),
                                                           ps.n_queries,
                                                           ps.n_rows,
                                                           ps.dim,
                                                           ps.k,
                                                           ps.metric);
        raft::update_host(
          distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
        raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
        raft::resource::sync_stream(handle_);
      }

      // Replace invalid edges with an edge to node 0
      auto graph_valid = raft::make_device_matrix<IdxT, int64_t>(
        handle_, index.graph().extent(0), index.graph().extent(1));
      raft::linalg::map(handle_, graph_valid.view(), edge_op{}, index.graph());

      auto cagra_index = cagra::index<DataT, IdxT>(handle_,
                                                   ps.metric,
                                                   raft::make_const_mdspan(database_view),
                                                   raft::make_const_mdspan(graph_valid.view()));

      cagra::search_params search_params;
      search_params.algo        = ps.algo;
      search_params.max_queries = ps.max_queries;
      search_params.team_size   = 0;

      rmm::device_uvector<DistanceT> distances_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_dev(queries_size, stream_);

      auto search_queries_view = raft::make_device_matrix_view<const DataT, int64_t>(
        search_queries.data(), ps.n_queries, ps.dim);
      auto indices_out_view =
        raft::make_device_matrix_view<IdxT, int64_t>(indices_dev.data(), ps.n_queries, ps.k);
      auto dists_out_view =
        raft::make_device_matrix_view<DistanceT, int64_t>(distances_dev.data(), ps.n_queries, ps.k);

      cagra::search(
        handle_, search_params, cagra_index, search_queries_view, indices_out_view, dists_out_view);
      raft::update_host(distances_Cagra.data(), distances_dev.data(), queries_size, stream_);
      raft::update_host(indices_Cagra.data(), indices_dev.data(), queries_size, stream_);

      raft::resource::sync_stream(handle_);

      double min_recall = ps.min_recall;
      EXPECT_TRUE(eval_neighbours(indices_naive,
                                  indices_Cagra,
                                  distances_naive,
                                  distances_Cagra,
                                  ps.n_queries,
                                  ps.k,
                                  0.003,
                                  min_recall));
    }
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    search_queries.resize(((size_t)ps.n_queries) * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      raft::random::normal(handle_, r, database.data(), ps.n_rows * ps.dim, DataT(0.1), DataT(2.0));
      raft::random::normal(
        handle_, r, search_queries.data(), ps.n_queries * ps.dim, DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.n_rows * ps.dim, DataT(1), DataT(20));
      raft::random::uniformInt(
        handle_, r, search_queries.data(), ps.n_queries * ps.dim, DataT(1), DataT(20));
    }
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnVamanaInputs ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

inline std::vector<AnnVamanaInputs> generate_inputs()
{
  std::vector<AnnVamanaInputs> inputs = raft::util::itertools::product<AnnVamanaInputs>(
    {1000},
    //    {1, 3, 5, 7, 8, 17, 64, 128, 137, 192, 256, 512, 619, 1024},  // TODO - fix alignment
    //    issue for odd dims
    {16, 32, 64, 128, 192, 256, 512, 1024},  // dim
    {32},                                    // graph degree
    {64, 128, 256},                          // visited_size
    {0.06, 0.1},
    {cuvs::distance::DistanceType::L2Expanded},
    {false},
    {100},
    {10},
    {cagra::search_algo::AUTO},
    {10},
    {64},
    {1},
    {0.2});

  std::vector<AnnVamanaInputs> inputs2 =
    raft::util::itertools::product<AnnVamanaInputs>({1000},
                                                    {16, 32, 64, 128, 192, 256, 512, 1024},  // dim
                                                    {64},             // graph degree
                                                    {128, 256, 512},  // visited_size
                                                    {0.06, 0.1},
                                                    {cuvs::distance::DistanceType::L2Expanded},
                                                    {false},
                                                    {100},
                                                    {10},
                                                    {cagra::search_algo::AUTO},
                                                    {10},
                                                    {32},
                                                    {1},
                                                    {0.2});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  inputs2 =
    raft::util::itertools::product<AnnVamanaInputs>({1000},
                                                    {16, 32, 64, 128, 192, 256, 512, 1024},  // dim
                                                    {128},       // graph degree
                                                    {256, 512},  // visited_size
                                                    {0.06, 0.1},
                                                    {cuvs::distance::DistanceType::L2Expanded},
                                                    {false},
                                                    {100},
                                                    {10},
                                                    {cagra::search_algo::AUTO},
                                                    {10},
                                                    {64},
                                                    {1},
                                                    {0.2});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  inputs2 =
    raft::util::itertools::product<AnnVamanaInputs>({1000},
                                                    {16, 32, 64, 128, 192, 256, 512, 1024},  // dim
                                                    {256},        // graph degree
                                                    {512, 1024},  // visited_size
                                                    {0.06, 0.1},
                                                    {cuvs::distance::DistanceType::L2Expanded},
                                                    {false},
                                                    {100},
                                                    {10},
                                                    {cagra::search_algo::AUTO},
                                                    {10},
                                                    {64},
                                                    {1},
                                                    {0.2});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  return inputs;
}

const std::vector<AnnVamanaInputs> inputs = generate_inputs();

}  // namespace cuvs::neighbors::experimental::vamana
