/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuvs/neighbors/ann_mg.hpp>

namespace cuvs::neighbors::mg {

enum class algo_t { IVF_FLAT, IVF_PQ, CAGRA };
enum class d_mode_t { REPLICATED, SHARDED, LOCAL_THEN_DISTRIBUTED };
enum class m_mode_t { MERGE_ON_ROOT_RANK, TREE_MERGE, UNDEFINED };

struct AnnMGInputs {
  int64_t num_queries;
  int64_t num_db_vecs;
  int64_t dim;
  int64_t k;
  d_mode_t d_mode;
  m_mode_t m_mode;
  algo_t algo;
  int64_t nprobe;
  int64_t nlist;
  cuvs::distance::DistanceType metric;
  bool adaptive_centers;
};

template <typename T, typename DataT>
class AnnMGTest : public ::testing::TestWithParam<AnnMGInputs> {
 public:
  AnnMGTest()
    : stream_(resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnMGInputs>::GetParam()),
      d_index_dataset(0, stream_),
      d_queries(0, stream_),
      h_index_dataset(0),
      h_queries(0)
  {
  }

  void testAnnMG()
  {
    size_t queries_size = ps.num_queries * ps.k;
    std::vector<int64_t> indices_naive(queries_size);
    std::vector<T> distances_naive(queries_size);
    std::vector<int64_t> indices_ann(queries_size);
    std::vector<T> distances_ann(queries_size);
    std::vector<uint32_t> indices_naive_32bits(queries_size);
    std::vector<uint32_t> indices_ann_32bits(queries_size);

    {
      rmm::device_uvector<T> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<int64_t> indices_naive_dev(queries_size, stream_);
      cuvs::neighbors::naive_knn<T, DataT, int64_t>(handle_,
                                                    distances_naive_dev.data(),
                                                    indices_naive_dev.data(),
                                                    d_queries.data(),
                                                    d_index_dataset.data(),
                                                    ps.num_queries,
                                                    ps.num_db_vecs,
                                                    ps.dim,
                                                    ps.k,
                                                    ps.metric);
      update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      resource::sync_stream(handle_);
    }

    int64_t n_rows_per_search_batch = 3000;  // [3000, 3000, 1000] == 7000 rows
    cuvs::neighbors::mg::nccl_clique clique;

    // IVF-Flat
    if (ps.algo == algo_t::IVF_FLAT &&
        (ps.d_mode == d_mode_t::REPLICATED || ps.d_mode == d_mode_t::SHARDED)) {
      distribution_mode d_mode;
      if (ps.d_mode == d_mode_t::REPLICATED)
        d_mode = distribution_mode::REPLICATED;
      else
        d_mode = distribution_mode::SHARDED;

      ivf_flat::mg_index_params index_params;
      index_params.n_lists                  = ps.nlist;
      index_params.metric                   = ps.metric;
      index_params.adaptive_centers         = ps.adaptive_centers;
      index_params.add_data_on_build        = false;
      index_params.kmeans_trainset_fraction = 1.0;
      index_params.metric_arg               = 0;
      index_params.mode                     = d_mode;

      ivf_flat::search_params search_params;
      search_params.n_probes = ps.nprobe;

      auto index_dataset = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_index_dataset.data(), ps.num_db_vecs, ps.dim);
      auto queries = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_queries.data(), ps.num_queries, ps.dim);
      auto neighbors = raft::make_host_matrix_view<int64_t, int64_t, row_major>(
        indices_ann.data(), ps.num_queries, ps.k);
      auto distances = raft::make_host_matrix_view<float, int64_t, row_major>(
        distances_ann.data(), ps.num_queries, ps.k);

      {
        auto index = cuvs::neighbors::mg::build(handle_, clique, index_params, index_dataset);
        cuvs::neighbors::mg::extend(handle_, clique, index, index_dataset, std::nullopt);
        cuvs::neighbors::mg::serialize(handle_, clique, index, "./cpp/build/ann_mg_ivf_flat_index");
      }
      auto new_index = cuvs::neighbors::mg::deserialize_flat<DataT, int64_t>(
        handle_, clique, "./cpp/build/ann_mg_ivf_flat_index");

      cuvs::neighbors::mg::sharded_merge_mode merge_mode;
      if (ps.m_mode == m_mode_t::MERGE_ON_ROOT_RANK)
        merge_mode = MERGE_ON_ROOT_RANK;
      else
        merge_mode = TREE_MERGE;
      cuvs::neighbors::mg::search(handle_,
                                  clique,
                                  new_index,
                                  search_params,
                                  queries,
                                  neighbors,
                                  distances,
                                  merge_mode,
                                  n_rows_per_search_batch);
      resource::sync_stream(handle_);

      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
      ASSERT_TRUE(eval_neighbours(indices_naive,
                                  indices_ann,
                                  distances_naive,
                                  distances_ann,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      std::fill(indices_ann.begin(), indices_ann.end(), 0);
      std::fill(distances_ann.begin(), distances_ann.end(), 0);
    }

    // IVF-PQ
    if (ps.algo == algo_t::IVF_PQ &&
        (ps.d_mode == d_mode_t::REPLICATED || ps.d_mode == d_mode_t::SHARDED)) {
      distribution_mode d_mode;
      if (ps.d_mode == d_mode_t::REPLICATED)
        d_mode = distribution_mode::REPLICATED;
      else
        d_mode = distribution_mode::SHARDED;

      ivf_pq::mg_index_params index_params;
      index_params.n_lists                  = ps.nlist;
      index_params.metric                   = ps.metric;
      index_params.add_data_on_build        = false;
      index_params.kmeans_trainset_fraction = 1.0;
      index_params.metric_arg               = 0;
      index_params.mode                     = d_mode;

      ivf_pq::search_params search_params;
      search_params.n_probes = ps.nprobe;

      auto index_dataset = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_index_dataset.data(), ps.num_db_vecs, ps.dim);
      auto queries = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_queries.data(), ps.num_queries, ps.dim);
      auto neighbors = raft::make_host_matrix_view<int64_t, int64_t, row_major>(
        indices_ann.data(), ps.num_queries, ps.k);
      auto distances = raft::make_host_matrix_view<float, int64_t, row_major>(
        distances_ann.data(), ps.num_queries, ps.k);

      {
        auto index = cuvs::neighbors::mg::build(handle_, clique, index_params, index_dataset);
        cuvs::neighbors::mg::extend(handle_, clique, index, index_dataset, std::nullopt);
        cuvs::neighbors::mg::serialize(handle_, clique, index, "./cpp/build/ann_mg_ivf_pq_index");
      }
      auto new_index = cuvs::neighbors::mg::deserialize_pq<DataT, int64_t>(
        handle_, clique, "./cpp/build/ann_mg_ivf_pq_index");

      cuvs::neighbors::mg::sharded_merge_mode merge_mode;
      if (ps.m_mode == m_mode_t::MERGE_ON_ROOT_RANK)
        merge_mode = MERGE_ON_ROOT_RANK;
      else
        merge_mode = TREE_MERGE;
      cuvs::neighbors::mg::search(handle_,
                                  clique,
                                  new_index,
                                  search_params,
                                  queries,
                                  neighbors,
                                  distances,
                                  merge_mode,
                                  n_rows_per_search_batch);
      resource::sync_stream(handle_);

      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
      ASSERT_TRUE(eval_neighbours(indices_naive,
                                  indices_ann,
                                  distances_naive,
                                  distances_ann,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      std::fill(indices_ann.begin(), indices_ann.end(), 0);
      std::fill(distances_ann.begin(), distances_ann.end(), 0);
    }

    // CAGRA
    if (ps.algo == algo_t::CAGRA &&
        (ps.d_mode == d_mode_t::REPLICATED || ps.d_mode == d_mode_t::SHARDED)) {
      distribution_mode d_mode;
      if (ps.d_mode == d_mode_t::REPLICATED)
        d_mode = distribution_mode::REPLICATED;
      else
        d_mode = distribution_mode::SHARDED;

      cagra::mg_index_params index_params;
      index_params.graph_build_params = cagra::graph_build_params::ivf_pq_params(
        raft::matrix_extent<int64_t>(ps.num_db_vecs, ps.dim));
      index_params.mode = d_mode;

      cagra::search_params search_params;

      auto index_dataset = raft::make_host_matrix_view<const DataT, uint32_t, row_major>(
        h_index_dataset.data(), ps.num_db_vecs, ps.dim);
      auto queries = raft::make_host_matrix_view<const DataT, uint32_t, row_major>(
        h_queries.data(), ps.num_queries, ps.dim);
      auto neighbors = raft::make_host_matrix_view<uint32_t, uint32_t, row_major>(
        indices_ann_32bits.data(), ps.num_queries, ps.k);
      auto distances = raft::make_host_matrix_view<float, uint32_t, row_major>(
        distances_ann.data(), ps.num_queries, ps.k);

      {
        auto index = cuvs::neighbors::mg::build(handle_, clique, index_params, index_dataset);
        cuvs::neighbors::mg::serialize(handle_, clique, index, "./cpp/build/ann_mg_cagra_index");
      }
      auto new_index = cuvs::neighbors::mg::deserialize_cagra<DataT, uint32_t>(
        handle_, clique, "./cpp/build/ann_mg_cagra_index");

      cuvs::neighbors::mg::sharded_merge_mode merge_mode;
      if (ps.m_mode == m_mode_t::MERGE_ON_ROOT_RANK)
        merge_mode = MERGE_ON_ROOT_RANK;
      else
        merge_mode = TREE_MERGE;
      cuvs::neighbors::mg::search(handle_,
                                  clique,
                                  new_index,
                                  search_params,
                                  queries,
                                  neighbors,
                                  distances,
                                  merge_mode,
                                  n_rows_per_search_batch);
      resource::sync_stream(handle_);

      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
      ASSERT_TRUE(eval_neighbours(indices_naive_32bits,
                                  indices_ann_32bits,
                                  distances_naive,
                                  distances_ann,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      std::fill(indices_ann_32bits.begin(), indices_ann_32bits.end(), 0);
      std::fill(distances_ann.begin(), distances_ann.end(), 0);
    }

    if (ps.algo == algo_t::IVF_FLAT && ps.d_mode == d_mode_t::LOCAL_THEN_DISTRIBUTED) {
      ivf_flat::index_params index_params;
      index_params.n_lists                  = ps.nlist;
      index_params.metric                   = ps.metric;
      index_params.adaptive_centers         = ps.adaptive_centers;
      index_params.add_data_on_build        = true;
      index_params.kmeans_trainset_fraction = 1.0;
      index_params.metric_arg               = 0;

      ivf_flat::search_params search_params;
      search_params.n_probes = ps.nprobe;

      RAFT_CUDA_TRY(cudaSetDevice(0));

      {
        auto index_dataset = raft::make_device_matrix_view<const DataT, int64_t>(
          d_index_dataset.data(), ps.num_db_vecs, ps.dim);
        auto index = cuvs::neighbors::ivf_flat::build(handle_, index_params, index_dataset);
        ivf_flat::serialize(handle_, "./cpp/build/local_ivf_flat_index", index);
      }

      auto queries = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_queries.data(), ps.num_queries, ps.dim);
      auto neighbors = raft::make_host_matrix_view<int64_t, int64_t, row_major>(
        indices_ann.data(), ps.num_queries, ps.k);
      auto distances = raft::make_host_matrix_view<float, int64_t, row_major>(
        distances_ann.data(), ps.num_queries, ps.k);

      auto distributed_index = cuvs::neighbors::mg::distribute_flat<DataT, int64_t>(
        handle_, clique, "./cpp/build/local_ivf_flat_index");
      cuvs::neighbors::mg::sharded_merge_mode merge_mode = TREE_MERGE;
      cuvs::neighbors::mg::search(handle_,
                                  clique,
                                  distributed_index,
                                  search_params,
                                  queries,
                                  neighbors,
                                  distances,
                                  merge_mode,
                                  n_rows_per_search_batch);

      resource::sync_stream(handle_);

      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
      ASSERT_TRUE(eval_neighbours(indices_naive,
                                  indices_ann,
                                  distances_naive,
                                  distances_ann,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      std::fill(indices_ann.begin(), indices_ann.end(), 0);
      std::fill(distances_ann.begin(), distances_ann.end(), 0);
    }

    if (ps.algo == algo_t::IVF_PQ && ps.d_mode == d_mode_t::LOCAL_THEN_DISTRIBUTED) {
      ivf_pq::index_params index_params;
      index_params.n_lists                  = ps.nlist;
      index_params.metric                   = ps.metric;
      index_params.add_data_on_build        = true;
      index_params.kmeans_trainset_fraction = 1.0;
      index_params.metric_arg               = 0;

      ivf_pq::search_params search_params;
      search_params.n_probes = ps.nprobe;

      RAFT_CUDA_TRY(cudaSetDevice(0));

      {
        auto index_dataset = raft::make_device_matrix_view<const DataT, int64_t>(
          d_index_dataset.data(), ps.num_db_vecs, ps.dim);
        auto index = cuvs::neighbors::ivf_pq::build(handle_, index_params, index_dataset);
        ivf_pq::serialize(handle_, "./cpp/build/local_ivf_pq_index", index);
      }

      auto queries = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_queries.data(), ps.num_queries, ps.dim);
      auto neighbors = raft::make_host_matrix_view<int64_t, int64_t, row_major>(
        indices_ann.data(), ps.num_queries, ps.k);
      auto distances = raft::make_host_matrix_view<float, int64_t, row_major>(
        distances_ann.data(), ps.num_queries, ps.k);

      auto distributed_index = cuvs::neighbors::mg::distribute_pq<DataT, int64_t>(
        handle_, clique, "./cpp/build/local_ivf_pq_index");
      cuvs::neighbors::mg::sharded_merge_mode merge_mode = TREE_MERGE;
      cuvs::neighbors::mg::search(handle_,
                                  clique,
                                  distributed_index,
                                  search_params,
                                  queries,
                                  neighbors,
                                  distances,
                                  merge_mode,
                                  n_rows_per_search_batch);

      resource::sync_stream(handle_);

      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
      ASSERT_TRUE(eval_neighbours(indices_naive,
                                  indices_ann,
                                  distances_naive,
                                  distances_ann,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      std::fill(indices_ann.begin(), indices_ann.end(), 0);
      std::fill(distances_ann.begin(), distances_ann.end(), 0);
    }

    if (ps.algo == algo_t::CAGRA && ps.d_mode == d_mode_t::LOCAL_THEN_DISTRIBUTED) {
      cagra::index_params index_params;
      index_params.graph_build_params = cagra::graph_build_params::ivf_pq_params(
        raft::matrix_extent<int64_t>(ps.num_db_vecs, ps.dim));

      cagra::search_params search_params;

      RAFT_CUDA_TRY(cudaSetDevice(0));

      {
        auto index_dataset = raft::make_device_matrix_view<const DataT, int64_t>(
          d_index_dataset.data(), ps.num_db_vecs, ps.dim);
        auto index = cuvs::neighbors::cagra::build(handle_, index_params, index_dataset);
        cuvs::neighbors::cagra::serialize(handle_, "./cpp/build/local_cagra_index", index);
      }

      auto queries = raft::make_host_matrix_view<const DataT, int64_t, row_major>(
        h_queries.data(), ps.num_queries, ps.dim);
      auto neighbors = raft::make_host_matrix_view<uint32_t, int64_t, row_major>(
        indices_ann_32bits.data(), ps.num_queries, ps.k);
      auto distances = raft::make_host_matrix_view<float, int64_t, row_major>(
        distances_ann.data(), ps.num_queries, ps.k);

      auto distributed_index = cuvs::neighbors::mg::distribute_cagra<DataT, uint32_t>(
        handle_, clique, "./cpp/build/local_cagra_index");

      cuvs::neighbors::mg::sharded_merge_mode merge_mode = TREE_MERGE;
      cuvs::neighbors::mg::search(handle_,
                                  clique,
                                  distributed_index,
                                  search_params,
                                  queries,
                                  neighbors,
                                  distances,
                                  merge_mode,
                                  n_rows_per_search_batch);

      resource::sync_stream(handle_);

      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);
      ASSERT_TRUE(eval_neighbours(indices_naive_32bits,
                                  indices_ann_32bits,
                                  distances_naive,
                                  distances_ann,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      std::fill(indices_ann_32bits.begin(), indices_ann_32bits.end(), 0);
      std::fill(distances_ann.begin(), distances_ann.end(), 0);
    }
  }

  void SetUp() override
  {
    d_index_dataset.resize(ps.num_db_vecs * ps.dim, stream_);
    d_queries.resize(ps.num_queries * ps.dim, stream_);
    h_index_dataset.resize(ps.num_db_vecs * ps.dim);
    h_queries.resize(ps.num_queries * ps.dim);

    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      raft::random::uniform(
        handle_, r, d_index_dataset.data(), d_index_dataset.size(), DataT(0.1), DataT(2.0));
      raft::random::uniform(handle_, r, d_queries.data(), d_queries.size(), DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, d_index_dataset.data(), d_index_dataset.size(), DataT(1), DataT(20));
      raft::random::uniformInt(handle_, r, d_queries.data(), d_queries.size(), DataT(1), DataT(20));
    }

    raft::copy(h_index_dataset.data(),
               d_index_dataset.data(),
               d_index_dataset.size(),
               resource::get_cuda_stream(handle_));
    raft::copy(
      h_queries.data(), d_queries.data(), d_queries.size(), resource::get_cuda_stream(handle_));
    resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    resource::sync_stream(handle_);
    h_index_dataset.clear();
    h_queries.clear();
    d_index_dataset.resize(0, stream_);
    d_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnMGInputs ps;
  std::vector<DataT> h_index_dataset;
  std::vector<DataT> h_queries;
  rmm::device_uvector<DataT> d_index_dataset;
  rmm::device_uvector<DataT> d_queries;
};

const std::vector<AnnMGInputs> inputs = {
  {7000,
   10000,
   8,
   16,
   d_mode_t::REPLICATED,
   m_mode_t::UNDEFINED,
   algo_t::IVF_FLAT,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::REPLICATED,
   m_mode_t::UNDEFINED,
   algo_t::IVF_PQ,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::REPLICATED,
   m_mode_t::UNDEFINED,
   algo_t::CAGRA,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},

  {7000,
   10000,
   8,
   16,
   d_mode_t::SHARDED,
   m_mode_t::MERGE_ON_ROOT_RANK,
   algo_t::IVF_FLAT,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::SHARDED,
   m_mode_t::MERGE_ON_ROOT_RANK,
   algo_t::IVF_PQ,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::SHARDED,
   m_mode_t::MERGE_ON_ROOT_RANK,
   algo_t::CAGRA,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},

  {7000,
   10000,
   8,
   16,
   d_mode_t::SHARDED,
   m_mode_t::TREE_MERGE,
   algo_t::IVF_FLAT,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::SHARDED,
   m_mode_t::TREE_MERGE,
   algo_t::IVF_PQ,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::SHARDED,
   m_mode_t::TREE_MERGE,
   algo_t::CAGRA,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},

  {7000,
   10000,
   8,
   16,
   d_mode_t::LOCAL_THEN_DISTRIBUTED,
   m_mode_t::UNDEFINED,
   algo_t::IVF_FLAT,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::LOCAL_THEN_DISTRIBUTED,
   m_mode_t::UNDEFINED,
   algo_t::IVF_PQ,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
  {7000,
   10000,
   8,
   16,
   d_mode_t::LOCAL_THEN_DISTRIBUTED,
   m_mode_t::UNDEFINED,
   algo_t::CAGRA,
   40,
   1024,
   cuvs::distance::DistanceType::L2Expanded,
   true},
};
}  // namespace cuvs::neighbors::mg
