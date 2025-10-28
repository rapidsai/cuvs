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

#include "ann_cagra.cuh"

#include <cuvs/neighbors/hnsw.hpp>

#include <cstdio>
#include <filesystem>

namespace cuvs::neighbors::cagra {

struct AnnCagraAceInputs {
  int n_queries;
  int n_rows;
  int dim;
  int k;
  int ace_npartitions;
  int ace_ef_construction;
  bool ace_use_disk;
  cuvs::distance::DistanceType metric;
  double min_recall;
};

inline ::std::ostream& operator<<(::std::ostream& os, const AnnCagraAceInputs& p)
{
  os << "{n_queries=" << p.n_queries << ", dataset shape=" << p.n_rows << "x" << p.dim
     << ", k=" << p.k << ", ace_npartitions=" << p.ace_npartitions
     << ", ace_ef_construction=" << p.ace_ef_construction
     << ", ace_use_disk=" << (p.ace_use_disk ? "true" : "false") << ", metric=";
  switch (p.metric) {
    case cuvs::distance::DistanceType::L2Expanded: os << "L2"; break;
    case cuvs::distance::DistanceType::InnerProduct: os << "InnerProduct"; break;
    default: os << "Unknown"; break;
  }
  os << ", min_recall=" << p.min_recall << "}";
  return os;
}

template <typename DistanceT, typename DataT, typename IdxT>
class AnnCagraAceTest : public ::testing::TestWithParam<AnnCagraAceInputs> {
 public:
  AnnCagraAceTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnCagraAceInputs>::GetParam()),
      database_dev(0, stream_),
      search_queries(0, stream_)
  {
  }

 protected:
  void testAce()
  {
    size_t queries_size = ps.n_queries * ps.k;
    std::vector<IdxT> indices_ace(queries_size);
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<DistanceT> distances_ace(queries_size);
    std::vector<DistanceT> distances_naive(queries_size);

    {
      rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);

      cuvs::neighbors::naive_knn<DistanceT, DataT, IdxT>(handle_,
                                                         distances_naive_dev.data(),
                                                         indices_naive_dev.data(),
                                                         search_queries.data(),
                                                         database_dev.data(),
                                                         ps.n_queries,
                                                         ps.n_rows,
                                                         ps.dim,
                                                         ps.k,
                                                         ps.metric);
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    // Create temporary directory for ACE build
    std::string temp_dir = std::string("/tmp/cuvs_ace_test_") + std::to_string(std::time(nullptr)) +
                           "_" + std::to_string(reinterpret_cast<uintptr_t>(this));
    std::filesystem::create_directories(temp_dir);

    {
      auto database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
      raft::copy(database_host.data_handle(), database_dev.data(), ps.n_rows * ps.dim, stream_);
      raft::resource::sync_stream(handle_);

      cagra::index_params index_params;
      index_params.metric                    = ps.metric;
      index_params.intermediate_graph_degree = 128;
      index_params.graph_degree              = 64;
      auto ace_params                        = graph_build_params::ace_params();
      ace_params.ace_npartitions             = ps.ace_npartitions;
      ace_params.ace_ef_construction         = ps.ace_ef_construction;
      ace_params.ace_build_dir               = temp_dir;
      ace_params.ace_use_disk                = ps.ace_use_disk;
      index_params.graph_build_params        = ace_params;

      auto index =
        cagra::build(handle_, index_params, raft::make_const_mdspan(database_host.view()));

      ASSERT_EQ(index.size(), ps.n_rows);

      if (ps.ace_use_disk) {
        // Verify disk-based ACE index using HNSW index from disk
        EXPECT_TRUE(index.on_disk());
        EXPECT_EQ(index.file_directory(), temp_dir);

        EXPECT_TRUE(std::filesystem::exists(temp_dir + "/cagra_graph.npy"));
        EXPECT_GE(std::filesystem::file_size(temp_dir + "/cagra_graph.npy"),
                  ps.n_rows * index_params.graph_degree * sizeof(IdxT));

        EXPECT_TRUE(std::filesystem::exists(temp_dir + "/reordered_dataset.npy"));
        EXPECT_GE(std::filesystem::file_size(temp_dir + "/reordered_dataset.npy"),
                  ps.n_rows * ps.dim * sizeof(DataT));

        EXPECT_TRUE(std::filesystem::exists(temp_dir + "/dataset_mapping.npy"));
        EXPECT_GE(std::filesystem::file_size(temp_dir + "/dataset_mapping.npy"),
                  ps.n_rows * sizeof(IdxT));

        hnsw::index_params hnsw_params;
        hnsw_params.hierarchy = hnsw::HnswHierarchy::GPU;

        auto hnsw_index = hnsw::from_cagra(handle_, hnsw_params, index);
        ASSERT_EQ(hnsw_index, nullptr);

        std::string hnsw_index_path = temp_dir + "/hnsw_index.bin";
        EXPECT_TRUE(std::filesystem::exists(hnsw_index_path));
        // For GPU hierarchy, HNSW index includes multi-layer structure
        // The size should be at least the base layer size
        auto hnsw_file_size = std::filesystem::file_size(hnsw_index_path);
        EXPECT_GE(hnsw_file_size, ps.n_rows * index_params.graph_degree * sizeof(IdxT));

        hnsw::index<DataT>* hnsw_index_raw = nullptr;
        hnsw::deserialize(
          handle_, hnsw_params, hnsw_index_path, ps.dim, ps.metric, &hnsw_index_raw);
        ASSERT_NE(hnsw_index_raw, nullptr);

        std::unique_ptr<hnsw::index<DataT>> hnsw_index_deserialized(hnsw_index_raw);
        EXPECT_EQ(hnsw_index_deserialized->dim(), ps.dim);
        EXPECT_EQ(hnsw_index_deserialized->metric(), ps.metric);

        auto queries_host = raft::make_host_matrix<DataT, int64_t>(ps.n_queries, ps.dim);
        raft::copy(
          queries_host.data_handle(), search_queries.data(), ps.n_queries * ps.dim, stream_);
        raft::resource::sync_stream(handle_);

        auto indices_hnsw_host   = raft::make_host_matrix<uint64_t, int64_t>(ps.n_queries, ps.k);
        auto distances_hnsw_host = raft::make_host_matrix<DistanceT, int64_t>(ps.n_queries, ps.k);

        hnsw::search_params search_params;
        search_params.ef          = std::max(ps.ace_ef_construction, ps.k * 2);
        search_params.num_threads = 1;

        hnsw::search(handle_,
                     search_params,
                     *hnsw_index_deserialized,
                     queries_host.view(),
                     indices_hnsw_host.view(),
                     distances_hnsw_host.view());

        for (size_t i = 0; i < queries_size; i++) {
          indices_ace[i]   = static_cast<IdxT>(indices_hnsw_host.data_handle()[i]);
          distances_ace[i] = distances_hnsw_host.data_handle()[i];
        }

        EXPECT_TRUE(eval_neighbours(indices_naive,
                                    indices_ace,
                                    distances_naive,
                                    distances_ace,
                                    ps.n_queries,
                                    ps.k,
                                    0.003,
                                    ps.min_recall))
          << "Disk-based ACE index loaded via HNSW failed recall check";
      } else {
        // For in-memory ACE, we can search directly
        EXPECT_FALSE(index.on_disk());
        ASSERT_GT(index.graph().size(), 0);
        EXPECT_EQ(index.graph_degree(), 64);

        rmm::device_uvector<DistanceT> distances_dev(queries_size, stream_);
        rmm::device_uvector<IdxT> indices_dev(queries_size, stream_);

        auto queries_view = raft::make_device_matrix_view<const DataT, int64_t>(
          search_queries.data(), ps.n_queries, ps.dim);
        auto indices_view =
          raft::make_device_matrix_view<IdxT, int64_t>(indices_dev.data(), ps.n_queries, ps.k);
        auto distances_view = raft::make_device_matrix_view<DistanceT, int64_t>(
          distances_dev.data(), ps.n_queries, ps.k);

        cagra::search_params search_params;
        search_params.itopk_size = 64;

        cagra::search(handle_, search_params, index, queries_view, indices_view, distances_view);

        raft::update_host(distances_ace.data(), distances_dev.data(), queries_size, stream_);
        raft::update_host(indices_ace.data(), indices_dev.data(), queries_size, stream_);
        raft::resource::sync_stream(handle_);

        EXPECT_TRUE(eval_neighbours(indices_naive,
                                    indices_ace,
                                    distances_naive,
                                    distances_ace,
                                    ps.n_queries,
                                    ps.k,
                                    0.003,
                                    ps.min_recall))
          << "In-memory ACE index failed recall check";
      }
    }

    // Clean up temporary directory
    std::filesystem::remove_all(temp_dir);
  }

  void SetUp() override
  {
    database_dev.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    search_queries.resize(ps.n_queries * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    InitDataset(handle_, database_dev.data(), ps.n_rows, ps.dim, ps.metric, r);
    InitDataset(handle_, search_queries.data(), ps.n_queries, ps.dim, ps.metric, r);
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database_dev.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnCagraAceInputs ps;
  rmm::device_uvector<DataT> database_dev;
  rmm::device_uvector<DataT> search_queries;
};

inline std::vector<AnnCagraAceInputs> generate_ace_inputs()
{
  return raft::util::itertools::product<AnnCagraAceInputs>(
    {10},           // n_queries
    {5000},         // n_rows
    {64, 128},      // dim
    {10},           // k
    {2, 4},         // ace_npartitions
    {100},          // ace_ef_construction
    {false, true},  // ace_use_disk (test both modes)
    {cuvs::distance::DistanceType::L2Expanded,
     cuvs::distance::DistanceType::InnerProduct},  // metric
    {0.95}                                         // min_recall
  );
}

const std::vector<AnnCagraAceInputs> ace_inputs = generate_ace_inputs();

}  // namespace cuvs::neighbors::cagra
