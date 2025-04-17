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
#pragma once

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include "naive_knn.cuh"
#include <cstddef>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/all_neighbors.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/nccl_clique.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>
#include <rapids_logger/logger.hpp>
#include <rmm/device_uvector.hpp>
#include <vector>

namespace cuvs::neighbors::all_neighbors {

enum knn_build_algo { NN_DESCENT, IVF_PQ };

struct AllNeighborsInputs {
  knn_build_algo build_algo;
  std::tuple<double, size_t, size_t> recall_cluster_nearestcluster;
  int n_rows;
  int dim;
  int k;
  cuvs::distance::DistanceType metric;
  bool data_on_host;
};

inline ::std::ostream& operator<<(::std::ostream& os, const AllNeighborsInputs& p)
{
  os << "dataset shape=" << p.n_rows << "x" << p.dim << ", graph_degree=" << p.k
     << ", metric=" << static_cast<int>(p.metric)
     << ", clusters=" << std::get<1>(p.recall_cluster_nearestcluster)
     << ", num nearest clusters=" << std::get<2>(p.recall_cluster_nearestcluster) << std::endl;
  return os;
}

template <typename DistanceT, typename DataT, typename IdxT = int64_t>
void get_graphs(raft::resources& handle,
                rmm::device_uvector<DataT>& database,
                std::vector<IdxT>& indices_bf,
                std::vector<DistanceT>& distances_bf,
                std::vector<IdxT>& indices_allNN,
                std::vector<DistanceT>& distances_allNN,
                AllNeighborsInputs& ps,
                size_t queries_size)
{
  index_params params;
  params.n_clusters         = std::get<1>(ps.recall_cluster_nearestcluster);
  params.n_nearest_clusters = std::get<2>(ps.recall_cluster_nearestcluster);
  params.metric             = ps.metric;

  if (ps.build_algo == NN_DESCENT) {
    auto nn_descent_params                      = graph_build_params::nn_descent_params{};
    nn_descent_params.max_iterations            = 100;
    nn_descent_params.graph_degree              = ps.k;
    nn_descent_params.intermediate_graph_degree = ps.k * 2;
    params.graph_build_params                   = nn_descent_params;
  } else if (ps.build_algo == IVF_PQ) {
    auto ivfq_build_params = graph_build_params::ivf_pq_params{};
    // heuristically good ivfpq n_lists
    ivfq_build_params.build_params.n_lists = std::max(
      5u,
      static_cast<uint32_t>(ps.n_rows * params.n_nearest_clusters / (5000 * params.n_clusters)));
    params.graph_build_params = ivfq_build_params;
  }

  auto cuda_stream = raft::resource::get_cuda_stream(handle);
  {
    rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, cuda_stream);
    rmm::device_uvector<IdxT> indices_naive_dev(queries_size, cuda_stream);
    naive_knn<DistanceT, DataT, IdxT>(handle,
                                      distances_naive_dev.data(),
                                      indices_naive_dev.data(),
                                      database.data(),
                                      database.data(),
                                      ps.n_rows,
                                      ps.n_rows,
                                      ps.dim,
                                      ps.k,
                                      ps.metric);
    raft::update_host(indices_bf.data(), indices_naive_dev.data(), queries_size, cuda_stream);
    raft::update_host(distances_bf.data(), distances_naive_dev.data(), queries_size, cuda_stream);

    raft::resource::sync_stream(handle);
  }

  {
    auto index = [&]() {
      if (ps.data_on_host) {
        auto database_h = raft::make_host_matrix<DataT, IdxT>(ps.n_rows, ps.dim);
        raft::copy(database_h.data_handle(), database.data(), ps.n_rows * ps.dim, cuda_stream);

        return all_neighbors::build(handle,
                                    raft::make_const_mdspan(database_h.view()),
                                    static_cast<int64_t>(ps.k),
                                    params,
                                    true);

      } else {
        return all_neighbors::build(
          handle,
          raft::make_device_matrix_view<const DataT, IdxT>(database.data(), ps.n_rows, ps.dim),
          static_cast<int64_t>(ps.k),
          params,
          true);
      }
    }();

    raft::copy(indices_allNN.data(), index.graph().data_handle(), queries_size, cuda_stream);
    if (index.distances().has_value()) {
      raft::copy(
        distances_allNN.data(), index.distances().value().data_handle(), queries_size, cuda_stream);
    }
    raft::resource::sync_stream(handle);
  }
}

template <typename DistanceT, typename DataT, typename IdxT = int64_t>
class AllNeighborsBatchTest : public ::testing::TestWithParam<AllNeighborsInputs> {
 public:
  AllNeighborsBatchTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      clique_(raft::resource::get_nccl_clique(handle_)),
      ps(::testing::TestWithParam<AllNeighborsInputs>::GetParam()),
      database(0, stream_)
  {
  }

 protected:
  void run()
  {
    size_t queries_size = static_cast<size_t>(ps.n_rows) * static_cast<size_t>(ps.k);

    std::vector<IdxT> indices_allNN(queries_size);
    std::vector<DistanceT> distances_allNN(queries_size);
    std::vector<DistanceT> distances_bf(queries_size);
    std::vector<IdxT> indices_bf(queries_size);
    std::cout << "start running here\n";
    get_graphs(handle_,
               database,
               indices_bf,
               distances_bf,
               indices_allNN,
               distances_allNN,
               ps,
               queries_size);

    double min_recall = std::get<0>(ps.recall_cluster_nearestcluster);
    EXPECT_TRUE(eval_recall(indices_bf, indices_allNN, ps.n_rows, ps.k, 0.01, min_recall, true));
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    auto database_view =
      raft::make_device_matrix_view<float, IdxT>(database.data(), ps.n_rows, ps.dim);
    auto labels = raft::make_device_vector<IdxT, IdxT>(handle_, ps.n_rows);
    raft::random::make_blobs(handle_, database_view, labels.view());
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  raft::comms::nccl_clique clique_;
  AllNeighborsInputs ps;
  rmm::device_uvector<DataT> database;
};

template <typename DistanceT, typename DataT, typename IdxT = int64_t>
class AllNeighborsSingleTest : public ::testing::TestWithParam<AllNeighborsInputs> {
 public:
  AllNeighborsSingleTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AllNeighborsInputs>::GetParam()),
      database(0, stream_)
  {
  }

 protected:
  void run()
  {
    size_t queries_size = static_cast<size_t>(ps.n_rows) * static_cast<size_t>(ps.k);

    std::vector<IdxT> indices_allNN(queries_size);
    std::vector<DistanceT> distances_allNN(queries_size);
    std::vector<DistanceT> distances_bf(queries_size);
    std::vector<IdxT> indices_bf(queries_size);

    get_graphs(handle_,
               database,
               indices_bf,
               distances_bf,
               indices_allNN,
               distances_allNN,
               ps,
               queries_size);

    double min_recall = std::get<0>(ps.recall_cluster_nearestcluster);
    EXPECT_TRUE(eval_recall(indices_bf, indices_allNN, ps.n_rows, ps.k, 0.01, min_recall, true));
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    auto database_view =
      raft::make_device_matrix_view<float, IdxT>(database.data(), ps.n_rows, ps.dim);
    auto labels = raft::make_device_vector<IdxT, IdxT>(handle_, ps.n_rows);
    raft::random::make_blobs(handle_, database_view, labels.view());
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AllNeighborsInputs ps;
  rmm::device_uvector<DataT> database;
};

const std::vector<AllNeighborsInputs> inputsSingle =
  raft::util::itertools::product<AllNeighborsInputs>(
    {IVF_PQ, NN_DESCENT},
    {std::make_tuple(0.9, 1lu, 2lu)},  // min_recall, n_clusters, num_nearest_cluster
    {5000, 7151},                      // n_rows
    {64, 137},                         // dim
    {16, 23},                          // graph_degree
    {cuvs::distance::DistanceType::L2Expanded},
    {false, true}  // data on host
  );

const std::vector<AllNeighborsInputs> inputsBatch =
  raft::util::itertools::product<AllNeighborsInputs>(
    {IVF_PQ, NN_DESCENT},
    {
      std::make_tuple(0.9, 4lu, 2lu),
      std::make_tuple(0.9, 7lu, 2lu),
      std::make_tuple(0.9, 10lu, 2lu),
    },             // min_recall, n_clusters, num_nearest_cluster
    {5000, 7151},  // n_rows
    {64, 137},     // dim
    {16, 23},      // graph_degree
    {cuvs::distance::DistanceType::L2Expanded},
    {true}  // data on host
  );

}  // namespace cuvs::neighbors::all_neighbors
