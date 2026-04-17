/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "ann_cagra.cuh"

#include <cuvs/neighbors/hnsw.hpp>
#include <cuvs/util/file_io.hpp>

#include <raft/core/detail/mdspan_numpy_serializer.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace cuvs::neighbors::hnsw {

struct AnnHnswAceInputs {
  int n_queries;
  int n_rows;
  int dim;
  int k;
  int npartitions;
  int ef_construction;
  bool use_disk;
  cuvs::distance::DistanceType metric;
  double min_recall;
  double max_host_memory_gb = 0;  // 0 = use system default
  double max_gpu_memory_gb  = 0;  // 0 = use system default
};

inline ::std::ostream& operator<<(::std::ostream& os, const AnnHnswAceInputs& p)
{
  os << "{n_queries=" << p.n_queries << ", dataset shape=" << p.n_rows << "x" << p.dim
     << ", k=" << p.k << ", npartitions=" << p.npartitions
     << ", ef_construction=" << p.ef_construction
     << ", use_disk=" << (p.use_disk ? "true" : "false") << ", metric=";
  switch (p.metric) {
    case cuvs::distance::DistanceType::L2Expanded: os << "L2"; break;
    case cuvs::distance::DistanceType::InnerProduct: os << "InnerProduct"; break;
    default: os << "Unknown"; break;
  }
  os << ", min_recall=" << p.min_recall;
  if (p.max_host_memory_gb > 0) { os << ", max_host_memory_gb=" << p.max_host_memory_gb; }
  if (p.max_gpu_memory_gb > 0) { os << ", max_gpu_memory_gb=" << p.max_gpu_memory_gb; }
  os << "}";
  return os;
}

template <typename DistanceT, typename DataT, typename IdxT>
class AnnHnswAceTest : public ::testing::TestWithParam<AnnHnswAceInputs> {
 public:
  AnnHnswAceTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnHnswAceInputs>::GetParam()),
      database_dev(0, stream_),
      search_queries(0, stream_)
  {
  }

 protected:
  auto load_npy_1d(const std::string& path) -> std::vector<uint32_t>
  {
    cuvs::util::file_descriptor fd(path, O_RDONLY);
    std::ifstream stream(path, std::ios::binary);
    EXPECT_TRUE(stream.is_open());
    if (!stream.is_open()) { return {}; }
    auto header = raft::detail::numpy_serializer::read_header(stream);
    auto offset = stream.tellg();
    EXPECT_NE(offset, std::streampos(-1));
    if (offset == std::streampos(-1)) { return {}; }
    EXPECT_EQ(header.shape.size(), 1);
    if (header.shape.size() != 1) { return {}; }
    std::vector<uint32_t> data(header.shape[0]);
    cuvs::util::read_large_file(
      fd, data.data(), data.size() * sizeof(uint32_t), static_cast<size_t>(offset));
    return data;
  }

  auto load_npy_2d(const std::string& path) -> std::pair<std::vector<uint32_t>, std::vector<size_t>>
  {
    cuvs::util::file_descriptor fd(path, O_RDONLY);
    std::ifstream stream(path, std::ios::binary);
    EXPECT_TRUE(stream.is_open());
    if (!stream.is_open()) { return {{}, {}}; }
    auto header = raft::detail::numpy_serializer::read_header(stream);
    auto offset = stream.tellg();
    EXPECT_NE(offset, std::streampos(-1));
    if (offset == std::streampos(-1)) { return {{}, {}}; }
    EXPECT_EQ(header.shape.size(), 2);
    if (header.shape.size() != 2) { return {{}, {}}; }
    std::vector<size_t> shape{header.shape[0], header.shape[1]};
    std::vector<uint32_t> data(shape[0] * shape[1]);
    cuvs::util::read_large_file(
      fd, data.data(), data.size() * sizeof(uint32_t), static_cast<size_t>(offset));
    return {std::move(data), std::move(shape)};
  }

  template <typename T>
  auto load_npy_2d_typed(const std::string& path) -> std::pair<std::vector<T>, std::vector<size_t>>
  {
    cuvs::util::file_descriptor fd(path, O_RDONLY);
    std::ifstream stream(path, std::ios::binary);
    EXPECT_TRUE(stream.is_open());
    if (!stream.is_open()) { return {{}, {}}; }
    auto header = raft::detail::numpy_serializer::read_header(stream);
    auto offset = stream.tellg();
    EXPECT_NE(offset, std::streampos(-1));
    if (offset == std::streampos(-1)) { return {{}, {}}; }
    EXPECT_EQ(header.shape.size(), 2);
    if (header.shape.size() != 2) { return {{}, {}}; }
    std::vector<size_t> shape{header.shape[0], header.shape[1]};
    std::vector<T> data(shape[0] * shape[1]);
    cuvs::util::read_large_file(
      fd, data.data(), data.size() * sizeof(T), static_cast<size_t>(offset));
    return {std::move(data), std::move(shape)};
  }

  template <typename MatrixView>
  void verifyDiskArtifacts(const std::string& temp_dir, MatrixView original_dataset)
  {
    const auto mapping_path         = temp_dir + "/dataset_mapping.npy";
    const auto reordered_data_path  = temp_dir + "/reordered_dataset.npy";
    const auto reordered_graph_path = temp_dir + "/cagra_graph.npy";
    const auto original_graph_path  = temp_dir + "/cagra_graph_original_ids.npy";

    ASSERT_TRUE(std::filesystem::exists(mapping_path));
    ASSERT_TRUE(std::filesystem::exists(reordered_data_path));
    ASSERT_TRUE(std::filesystem::exists(reordered_graph_path));
    ASSERT_TRUE(std::filesystem::exists(original_graph_path));

    auto mapping = load_npy_1d(mapping_path);
    auto [reordered_dataset, reordered_dataset_shape] =
      load_npy_2d_typed<DataT>(reordered_data_path);
    auto [reordered_graph, reordered_shape] = load_npy_2d(reordered_graph_path);
    auto [original_graph, original_shape]   = load_npy_2d(original_graph_path);

    ASSERT_EQ(reordered_dataset_shape.size(), 2);
    ASSERT_EQ(reordered_dataset_shape[0], mapping.size());
    ASSERT_EQ(reordered_dataset_shape[0], static_cast<size_t>(original_dataset.extent(0)));
    ASSERT_EQ(reordered_dataset_shape[1], static_cast<size_t>(original_dataset.extent(1)));

    const auto dim = reordered_dataset_shape[1];
    std::vector<uint8_t> seen_original_rows(mapping.size(), 0);
    for (size_t reordered_row = 0; reordered_row < mapping.size(); ++reordered_row) {
      const auto original_row = mapping[reordered_row];
      ASSERT_LT(static_cast<size_t>(original_row), static_cast<size_t>(original_dataset.extent(0)));
      ASSERT_EQ(seen_original_rows[original_row], 0)
        << "Mapping is not a permutation; original_row=" << original_row
        << " appears more than once";
      seen_original_rows[original_row] = 1;

      const auto* reordered_row_ptr = reordered_dataset.data() + reordered_row * dim;
      const auto* original_row_ptr =
        original_dataset.data_handle() + static_cast<size_t>(original_row) * dim;
      ASSERT_EQ(std::memcmp(reordered_row_ptr, original_row_ptr, dim * sizeof(DataT)), 0)
        << "Reordered dataset mismatch at reordered_row=" << reordered_row
        << ", original_row=" << original_row;
    }
    for (size_t original_row = 0; original_row < seen_original_rows.size(); ++original_row) {
      ASSERT_EQ(seen_original_rows[original_row], 1)
        << "Mapping is not onto; original_row=" << original_row << " was never referenced";
    }

    ASSERT_EQ(reordered_shape, original_shape);
    ASSERT_EQ(reordered_shape[0], mapping.size());

    const auto n_rows       = reordered_shape[0];
    const auto graph_degree = reordered_shape[1];

    for (size_t reordered_row = 0; reordered_row < n_rows; ++reordered_row) {
      const auto original_row = mapping[reordered_row];
      for (size_t neighbor_idx = 0; neighbor_idx < graph_degree; ++neighbor_idx) {
        const auto reordered_neighbor =
          reordered_graph[reordered_row * graph_degree + neighbor_idx];
        ASSERT_LT(static_cast<size_t>(reordered_neighbor), mapping.size());
        const auto expected_original_neighbor = mapping[reordered_neighbor];
        const auto actual_original_neighbor =
          original_graph[original_row * graph_degree + neighbor_idx];
        ASSERT_EQ(actual_original_neighbor, expected_original_neighbor)
          << "Remapped graph mismatch at reordered_row=" << reordered_row
          << ", original_row=" << original_row << ", neighbor_idx=" << neighbor_idx;
      }
    }
  }

  void testHnswAceBuild()
  {
    size_t queries_size = ps.n_queries * ps.k;
    std::vector<uint64_t> indexes_hnsw(queries_size);
    std::vector<IdxT> indexes_naive(queries_size);
    std::vector<DistanceT> distances_hnsw(queries_size);
    std::vector<DistanceT> distances_naive(queries_size);

    {
      rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indexes_naive_dev(queries_size, stream_);

      cuvs::neighbors::naive_knn<DistanceT, DataT, IdxT>(handle_,
                                                         distances_naive_dev.data(),
                                                         indexes_naive_dev.data(),
                                                         search_queries.data(),
                                                         database_dev.data(),
                                                         ps.n_queries,
                                                         ps.n_rows,
                                                         ps.dim,
                                                         ps.k,
                                                         ps.metric);
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::update_host(indexes_naive.data(), indexes_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    // Create temporary directory for ACE build
    std::string temp_dir = std::string("/tmp/cuvs_hnsw_ace_test_") +
                           std::to_string(std::time(nullptr)) + "_" +
                           std::to_string(reinterpret_cast<uintptr_t>(this));
    std::filesystem::create_directories(temp_dir);

    {
      // Copy dataset to host for hnsw::build
      auto database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
      raft::copy(database_host.data_handle(), database_dev.data(), ps.n_rows * ps.dim, stream_);
      raft::resource::sync_stream(handle_);

      // Configure HNSW index parameters with ACE
      hnsw::index_params hnsw_params;
      hnsw_params.metric          = ps.metric;
      hnsw_params.hierarchy       = hnsw::HnswHierarchy::GPU;
      hnsw_params.M               = 32;
      hnsw_params.ef_construction = ps.ef_construction;

      // Configure ACE parameters
      auto ace_params                = graph_build_params::ace_params();
      ace_params.npartitions         = ps.npartitions;
      ace_params.build_dir           = temp_dir;
      ace_params.use_disk            = ps.use_disk;
      ace_params.max_host_memory_gb  = ps.max_host_memory_gb;
      ace_params.max_gpu_memory_gb   = ps.max_gpu_memory_gb;
      hnsw_params.graph_build_params = ace_params;

      // Build HNSW index using ACE
      auto hnsw_index =
        hnsw::build(handle_, hnsw_params, raft::make_const_mdspan(database_host.view()));

      ASSERT_NE(hnsw_index, nullptr);
      if (ps.use_disk) { verifyDiskArtifacts(temp_dir, database_host.view()); }

      // Prepare queries on host
      auto queries_host = raft::make_host_matrix<DataT, int64_t>(ps.n_queries, ps.dim);
      raft::copy(queries_host.data_handle(), search_queries.data(), ps.n_queries * ps.dim, stream_);
      raft::resource::sync_stream(handle_);

      auto indexes_hnsw_host   = raft::make_host_matrix<uint64_t, int64_t>(ps.n_queries, ps.k);
      auto distances_hnsw_host = raft::make_host_matrix<DistanceT, int64_t>(ps.n_queries, ps.k);

      // Search the HNSW index
      hnsw::search_params search_params;
      search_params.ef          = std::max(ps.ef_construction, ps.k * 2);
      search_params.num_threads = 1;

      if (!ps.use_disk) {
        hnsw::search(handle_,
                     search_params,
                     *hnsw_index,
                     queries_host.view(),
                     indexes_hnsw_host.view(),
                     distances_hnsw_host.view());
        for (size_t i = 0; i < queries_size; i++) {
          indexes_hnsw[i]   = indexes_hnsw_host.data_handle()[i];
          distances_hnsw[i] = distances_hnsw_host.data_handle()[i];
        }

        // Convert indexes for comparison
        std::vector<IdxT> indexes_hnsw_converted(queries_size);
        for (size_t i = 0; i < queries_size; i++) {
          indexes_hnsw_converted[i] = static_cast<IdxT>(indexes_hnsw[i]);
        }

        EXPECT_TRUE(cuvs::neighbors::eval_neighbours(indexes_naive,
                                                     indexes_hnsw_converted,
                                                     distances_naive,
                                                     distances_hnsw,
                                                     ps.n_queries,
                                                     ps.k,
                                                     0.003,
                                                     ps.min_recall))
          << "HNSW ACE build and search failed recall check";
      }

      tmp_index_file index_file;
      hnsw::serialize(handle_, index_file.filename, *hnsw_index);

      hnsw::index<DataT>* deserialized_index = nullptr;
      hnsw::deserialize(
        handle_, hnsw_params, index_file.filename, ps.dim, ps.metric, &deserialized_index);
      ASSERT_NE(deserialized_index, nullptr);

      // Reset search results
      for (size_t i = 0; i < queries_size; i++) {
        indexes_hnsw[i]   = 0;
        distances_hnsw[i] = 0;
      }
      hnsw::search(handle_,
                   search_params,
                   *deserialized_index,
                   queries_host.view(),
                   indexes_hnsw_host.view(),
                   distances_hnsw_host.view());
      for (size_t i = 0; i < queries_size; i++) {
        indexes_hnsw[i]   = indexes_hnsw_host.data_handle()[i];
        distances_hnsw[i] = distances_hnsw_host.data_handle()[i];
      }

      // Convert indexes for comparison
      std::vector<IdxT> indexes_hnsw_converted(queries_size);
      for (size_t i = 0; i < queries_size; i++) {
        indexes_hnsw_converted[i] = static_cast<IdxT>(indexes_hnsw[i]);
      }

      EXPECT_TRUE(cuvs::neighbors::eval_neighbours(indexes_naive,
                                                   indexes_hnsw_converted,
                                                   distances_naive,
                                                   distances_hnsw,
                                                   ps.n_queries,
                                                   ps.k,
                                                   0.003,
                                                   ps.min_recall))
        << "HNSW ACE build and search failed recall check";
      // Clean up deserialized index
      delete deserialized_index;
    }

    // Clean up temporary directory
    std::filesystem::remove_all(temp_dir);
  }

  void testHnswAceMemoryLimitFallback()
  {
    // This test verifies that setting tiny memory limits forces disk mode automatically
    // Create temporary directory for ACE build
    std::string temp_dir = std::string("/tmp/cuvs_hnsw_ace_memlimit_test_") +
                           std::to_string(std::time(nullptr)) + "_" +
                           std::to_string(reinterpret_cast<uintptr_t>(this));
    std::filesystem::create_directories(temp_dir);

    {
      // Copy dataset to host for hnsw::build
      auto database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
      raft::copy(database_host.data_handle(), database_dev.data(), ps.n_rows * ps.dim, stream_);
      raft::resource::sync_stream(handle_);

      // Configure HNSW index parameters with ACE
      hnsw::index_params hnsw_params;
      hnsw_params.metric    = ps.metric;
      hnsw_params.hierarchy = hnsw::HnswHierarchy::GPU;
      hnsw_params.M         = 32;

      // Configure ACE parameters with tiny memory limits to force disk mode
      auto ace_params            = graph_build_params::ace_params();
      ace_params.npartitions     = ps.npartitions;
      ace_params.ef_construction = ps.ef_construction;
      ace_params.build_dir       = temp_dir;
      ace_params.use_disk        = false;  // Not explicitly requesting disk mode
      // Set tiny memory limits (0.001 GiB = ~1 MB) to force disk mode
      ace_params.max_host_memory_gb  = 0.001;
      ace_params.max_gpu_memory_gb   = 0.001;
      hnsw_params.graph_build_params = ace_params;

      // Build HNSW index using ACE - should automatically fall back to disk mode
      auto hnsw_index =
        hnsw::build(handle_, hnsw_params, raft::make_const_mdspan(database_host.view()));

      ASSERT_NE(hnsw_index, nullptr);

      // Verify that disk mode was triggered by checking for the expected files
      std::string graph_file     = temp_dir + "/cagra_graph.npy";
      std::string reordered_file = temp_dir + "/reordered_dataset.npy";

      EXPECT_TRUE(std::filesystem::exists(graph_file))
        << "Graph file should exist when memory limit triggers disk mode fallback";
      EXPECT_TRUE(std::filesystem::exists(reordered_file))
        << "Reordered dataset file should exist when memory limit triggers disk mode fallback";
      EXPECT_TRUE(std::filesystem::exists(temp_dir + "/cagra_graph_original_ids.npy"))
        << "Original-id remapped graph file should exist when disk mode is used";
      verifyDiskArtifacts(temp_dir, database_host.view());
    }

    // Clean up temporary directory
    std::filesystem::remove_all(temp_dir);
  }

  void SetUp() override
  {
    database_dev.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    search_queries.resize(ps.n_queries * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    cuvs::neighbors::cagra::InitDataset(
      handle_, database_dev.data(), ps.n_rows, ps.dim, ps.metric, r);
    cuvs::neighbors::cagra::InitDataset(
      handle_, search_queries.data(), ps.n_queries, ps.dim, ps.metric, r);
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
  AnnHnswAceInputs ps;
  rmm::device_uvector<DataT> database_dev;
  rmm::device_uvector<DataT> search_queries;
};

inline std::vector<AnnHnswAceInputs> generate_hnsw_ace_inputs()
{
  return raft::util::itertools::product<AnnHnswAceInputs>(
    {10},           // n_queries
    {5000},         // n_rows
    {64, 128},      // dim
    {10},           // k
    {2, 4},         // npartitions
    {100},          // ef_construction
    {false, true},  // use_disk (test both modes)
    {cuvs::distance::DistanceType::L2Expanded,
     cuvs::distance::DistanceType::InnerProduct},  // metric
    {0.9},                                         // min_recall
    {0.0},                                         // max_host_memory_gb (0 = use default)
    {0.0}                                          // max_gpu_memory_gb (0 = use default)
  );
}

// Inputs specifically for testing memory limit fallback to disk mode
inline std::vector<AnnHnswAceInputs> generate_hnsw_ace_memory_fallback_inputs()
{
  return {         // Test with L2 metric
          {10,     // n_queries
           5000,   // n_rows
           64,     // dim
           10,     // k
           2,      // npartitions
           100,    // ef_construction
           false,  // use_disk (not explicitly set, should be triggered by memory limit)
           cuvs::distance::DistanceType::L2Expanded,
           0.0,     // min_recall (not checked in fallback test)
           0.001,   // max_host_memory_gb (tiny limit to force disk mode)
           0.001},  // max_gpu_memory_gb (tiny limit to force disk mode)
          // Regression case: enough rows to force a short final remap batch.
          {10,
           360000,
           16,
           10,
           4,
           100,
           false,
           cuvs::distance::DistanceType::L2Expanded,
           0.0,
           0.001,
           0.001}};
}

const std::vector<AnnHnswAceInputs> hnsw_ace_inputs = generate_hnsw_ace_inputs();
const std::vector<AnnHnswAceInputs> hnsw_ace_memory_fallback_inputs =
  generate_hnsw_ace_memory_fallback_inputs();

}  // namespace cuvs::neighbors::hnsw
