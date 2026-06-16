/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "ann_cagra.cuh"

#include <cuvs/neighbors/hnsw.hpp>
#include <cuvs/util/file_io.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <vector>

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
    }

    // Clean up temporary directory
    std::filesystem::remove_all(temp_dir);
  }

  void testHnswAceLayeredBuildDeserializeSearch()
  {
    size_t queries_size = ps.n_queries * ps.k;
    std::vector<IdxT> indexes_naive(queries_size);
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

    std::string temp_dir = std::string("/tmp/cuvs_hnsw_ace_layered_test_") +
                           std::to_string(std::time(nullptr)) + "_" +
                           std::to_string(reinterpret_cast<uintptr_t>(this));
    std::filesystem::create_directories(temp_dir);
    struct temp_dir_cleanup {
      std::string path;
      ~temp_dir_cleanup()
      {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
      }
    } cleanup{temp_dir};

    auto database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
    raft::copy(database_host.data_handle(), database_dev.data(), ps.n_rows * ps.dim, stream_);
    auto queries_host = raft::make_host_matrix<DataT, int64_t>(ps.n_queries, ps.dim);
    raft::copy(queries_host.data_handle(), search_queries.data(), ps.n_queries * ps.dim, stream_);
    raft::resource::sync_stream(handle_);

    const auto dataset_file = (std::filesystem::path(temp_dir) / "dataset.npy").string();
    auto [dataset_fd, dataset_header_size] = cuvs::util::create_numpy_file<DataT>(
      dataset_file, {static_cast<size_t>(ps.n_rows), static_cast<size_t>(ps.dim)});
    cuvs::util::write_large_file(dataset_fd,
                                 database_host.data_handle(),
                                 static_cast<size_t>(ps.n_rows) * ps.dim * sizeof(DataT),
                                 dataset_header_size);

    hnsw::index_params hnsw_params;
    hnsw_params.metric          = ps.metric;
    hnsw_params.hierarchy       = hnsw::HnswHierarchy::GPU_LAYERED_ON_DISK;
    hnsw_params.M               = 32;
    hnsw_params.ef_construction = ps.ef_construction;
    hnsw_params.dataset_path    = dataset_file;

    auto ace_params                = graph_build_params::ace_params();
    ace_params.npartitions         = ps.npartitions;
    ace_params.build_dir           = temp_dir;
    ace_params.use_disk            = true;
    ace_params.max_host_memory_gb  = ps.max_host_memory_gb;
    ace_params.max_gpu_memory_gb   = ps.max_gpu_memory_gb;
    hnsw_params.graph_build_params = ace_params;

    auto hnsw_index =
      hnsw::build(handle_, hnsw_params, raft::make_const_mdspan(database_host.view()));
    ASSERT_NE(hnsw_index, nullptr);

    const auto artifact_path = hnsw_index->file_path();
    ASSERT_FALSE(artifact_path.empty());
    ASSERT_TRUE(std::filesystem::is_regular_file(artifact_path));
    EXPECT_EQ(std::filesystem::path(artifact_path).filename().string(), "hnsw_index.cuvs");
    size_t cuvs_artifact_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(temp_dir)) {
      if (entry.path().extension() == ".cuvs") { ++cuvs_artifact_count; }
    }
    EXPECT_EQ(cuvs_artifact_count, 1);
    EXPECT_FALSE(std::filesystem::exists(std::filesystem::path(temp_dir) / "layered_hnsw"));

    auto indexes_hnsw_host   = raft::make_host_matrix<uint64_t, int64_t>(ps.n_queries, ps.k);
    auto distances_hnsw_host = raft::make_host_matrix<DistanceT, int64_t>(ps.n_queries, ps.k);

    hnsw::search_params search_params;
    search_params.ef          = std::max(ps.ef_construction, ps.k * 2);
    search_params.num_threads = 1;

    EXPECT_THROW(hnsw::search(handle_,
                              search_params,
                              *hnsw_index,
                              queries_host.view(),
                              indexes_hnsw_host.view(),
                              distances_hnsw_host.view()),
                 std::exception);

    hnsw::index<DataT>* deserialized_index = nullptr;
    hnsw::deserialize(handle_, hnsw_params, artifact_path, ps.dim, ps.metric, &deserialized_index);
    ASSERT_NE(deserialized_index, nullptr);
    std::unique_ptr<hnsw::index<DataT>> deserialized_guard(deserialized_index);

    hnsw::search(handle_,
                 search_params,
                 *deserialized_guard,
                 queries_host.view(),
                 indexes_hnsw_host.view(),
                 distances_hnsw_host.view());

    std::vector<IdxT> indexes_hnsw_converted(queries_size);
    std::vector<DistanceT> distances_hnsw(queries_size);
    for (size_t i = 0; i < queries_size; i++) {
      indexes_hnsw_converted[i] = static_cast<IdxT>(indexes_hnsw_host.data_handle()[i]);
      distances_hnsw[i]         = distances_hnsw_host.data_handle()[i];
    }

    EXPECT_TRUE(cuvs::neighbors::eval_neighbours(indexes_naive,
                                                 indexes_hnsw_converted,
                                                 distances_naive,
                                                 distances_hnsw,
                                                 ps.n_queries,
                                                 ps.k,
                                                 0.003,
                                                 ps.min_recall))
      << "Layered HNSW deserialize and search failed recall check";

    const auto copied_artifact =
      (std::filesystem::path(temp_dir) / "copied_layered" / "hnsw_index.cuvs").string();
    hnsw::serialize(handle_, copied_artifact, *hnsw_index);
    EXPECT_TRUE(std::filesystem::is_regular_file(copied_artifact));
    size_t copied_file_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(
           std::filesystem::path(copied_artifact).parent_path())) {
      if (entry.is_regular_file()) { ++copied_file_count; }
    }
    EXPECT_EQ(copied_file_count, 1);

    hnsw::index<DataT>* copied_index = nullptr;
    hnsw::deserialize(handle_, hnsw_params, copied_artifact, ps.dim, ps.metric, &copied_index);
    ASSERT_NE(copied_index, nullptr);
    std::unique_ptr<hnsw::index<DataT>> copied_guard(copied_index);

    hnsw::search(handle_,
                 search_params,
                 *copied_guard,
                 queries_host.view(),
                 indexes_hnsw_host.view(),
                 distances_hnsw_host.view());

    for (size_t i = 0; i < queries_size; i++) {
      indexes_hnsw_converted[i] = static_cast<IdxT>(indexes_hnsw_host.data_handle()[i]);
      distances_hnsw[i]         = distances_hnsw_host.data_handle()[i];
    }
    EXPECT_TRUE(cuvs::neighbors::eval_neighbours(indexes_naive,
                                                 indexes_hnsw_converted,
                                                 distances_naive,
                                                 distances_hnsw,
                                                 ps.n_queries,
                                                 ps.k,
                                                 0.003,
                                                 ps.min_recall))
      << "Copied layered HNSW artifact deserialize and search failed recall check";

    const auto bad_artifact =
      (std::filesystem::path(temp_dir) / "bad_layered" / "bad_magic.cuvs").string();
    std::filesystem::create_directories(std::filesystem::path(bad_artifact).parent_path());
    {
      std::ofstream bad_file(bad_artifact, std::ios::binary);
      std::array<char, 64> bad_bytes{};
      bad_file.write(bad_bytes.data(), bad_bytes.size());
    }
    hnsw::index<DataT>* bad_index = nullptr;
    EXPECT_THROW(
      hnsw::deserialize(handle_, hnsw_params, bad_artifact, ps.dim, ps.metric, &bad_index),
      std::exception);

    const auto bad_version_artifact =
      (std::filesystem::path(temp_dir) / "bad_layered" / "bad_version.cuvs").string();
    std::filesystem::copy_file(
      copied_artifact, bad_version_artifact, std::filesystem::copy_options::overwrite_existing);
    constexpr std::streamoff layered_header_version_offset = 32;
    {
      std::fstream bad_version_file(bad_version_artifact,
                                    std::ios::in | std::ios::out | std::ios::binary);
      const uint32_t bad_version = 999;
      bad_version_file.seekp(layered_header_version_offset);
      bad_version_file.write(reinterpret_cast<const char*>(&bad_version), sizeof(bad_version));
    }
    hnsw::index<DataT>* bad_version_index = nullptr;
    EXPECT_THROW(
      hnsw::deserialize(
        handle_, hnsw_params, bad_version_artifact, ps.dim, ps.metric, &bad_version_index),
      std::exception);

    auto missing_dataset_params = hnsw_params;
    missing_dataset_params.dataset_path.clear();
    hnsw::index<DataT>* missing_dataset_index = nullptr;
    EXPECT_THROW(hnsw::deserialize(handle_,
                                   missing_dataset_params,
                                   copied_artifact,
                                   ps.dim,
                                   ps.metric,
                                   &missing_dataset_index),
                 std::exception);

    const auto truncated_artifact =
      (std::filesystem::path(temp_dir) / "bad_layered" / "truncated.cuvs").string();
    std::filesystem::copy_file(
      copied_artifact, truncated_artifact, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::resize_file(truncated_artifact, 128);
    hnsw::index<DataT>* truncated_index = nullptr;
    EXPECT_THROW(hnsw::deserialize(
                   handle_, hnsw_params, truncated_artifact, ps.dim, ps.metric, &truncated_index),
                 std::exception);

    const auto wrong_dataset_file =
      (std::filesystem::path(temp_dir) / "bad_layered" / "wrong_dataset.npy").string();
    auto [wrong_dataset_fd, wrong_dataset_header_size] = cuvs::util::create_numpy_file<DataT>(
      wrong_dataset_file, {static_cast<size_t>(ps.n_rows - 1), static_cast<size_t>(ps.dim)});
    cuvs::util::write_large_file(wrong_dataset_fd,
                                 database_host.data_handle(),
                                 static_cast<size_t>(ps.n_rows - 1) * ps.dim * sizeof(DataT),
                                 wrong_dataset_header_size);
    auto wrong_dataset_params               = hnsw_params;
    wrong_dataset_params.dataset_path       = wrong_dataset_file;
    hnsw::index<DataT>* wrong_dataset_index = nullptr;
    EXPECT_THROW(
      hnsw::deserialize(
        handle_, wrong_dataset_params, copied_artifact, ps.dim, ps.metric, &wrong_dataset_index),
      std::exception);
  }

  void testHnswAceLayeredMaterializeToHnswlib()
  {
    size_t queries_size = ps.n_queries * ps.k;
    std::vector<IdxT> indexes_naive(queries_size);
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

    std::string temp_dir = std::string("/tmp/cuvs_hnsw_ace_materialize_test_") +
                           std::to_string(std::time(nullptr)) + "_" +
                           std::to_string(reinterpret_cast<uintptr_t>(this));
    std::filesystem::create_directories(temp_dir);
    struct temp_dir_cleanup {
      std::string path;
      ~temp_dir_cleanup()
      {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
      }
    } cleanup{temp_dir};

    auto database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
    raft::copy(database_host.data_handle(), database_dev.data(), ps.n_rows * ps.dim, stream_);
    auto queries_host = raft::make_host_matrix<DataT, int64_t>(ps.n_queries, ps.dim);
    raft::copy(queries_host.data_handle(), search_queries.data(), ps.n_queries * ps.dim, stream_);
    raft::resource::sync_stream(handle_);

    const auto dataset_file = (std::filesystem::path(temp_dir) / "dataset.npy").string();
    auto [dataset_fd, dataset_header_size] = cuvs::util::create_numpy_file<DataT>(
      dataset_file, {static_cast<size_t>(ps.n_rows), static_cast<size_t>(ps.dim)});
    cuvs::util::write_large_file(dataset_fd,
                                 database_host.data_handle(),
                                 static_cast<size_t>(ps.n_rows) * ps.dim * sizeof(DataT),
                                 dataset_header_size);

    hnsw::index_params hnsw_params;
    hnsw_params.metric          = ps.metric;
    hnsw_params.hierarchy       = hnsw::HnswHierarchy::GPU_LAYERED_ON_DISK;
    hnsw_params.M               = 32;
    hnsw_params.ef_construction = ps.ef_construction;
    hnsw_params.dataset_path    = dataset_file;

    auto ace_params                = graph_build_params::ace_params();
    ace_params.npartitions         = ps.npartitions;
    ace_params.build_dir           = temp_dir;
    ace_params.use_disk            = true;
    ace_params.max_host_memory_gb  = ps.max_host_memory_gb;
    ace_params.max_gpu_memory_gb   = ps.max_gpu_memory_gb;
    hnsw_params.graph_build_params = ace_params;

    auto hnsw_index =
      hnsw::build(handle_, hnsw_params, raft::make_const_mdspan(database_host.view()));
    ASSERT_NE(hnsw_index, nullptr);
    const auto artifact_path = hnsw_index->file_path();
    ASSERT_FALSE(artifact_path.empty());

    hnsw::search_params search_params;
    search_params.ef          = std::max(ps.ef_construction, ps.k * 2);
    search_params.num_threads = 1;

    auto indexes_hnsw_host   = raft::make_host_matrix<uint64_t, int64_t>(ps.n_queries, ps.k);
    auto distances_hnsw_host = raft::make_host_matrix<DistanceT, int64_t>(ps.n_queries, ps.k);

    // Reference: load the layered artifact in RAM and search.
    std::vector<IdxT> indexes_layered(queries_size);
    std::vector<DistanceT> distances_layered(queries_size);
    {
      hnsw::index<DataT>* layered_index = nullptr;
      hnsw::deserialize(handle_, hnsw_params, artifact_path, ps.dim, ps.metric, &layered_index);
      ASSERT_NE(layered_index, nullptr);
      std::unique_ptr<hnsw::index<DataT>> layered_guard(layered_index);
      hnsw::search(handle_,
                   search_params,
                   *layered_guard,
                   queries_host.view(),
                   indexes_hnsw_host.view(),
                   distances_hnsw_host.view());
      for (size_t i = 0; i < queries_size; i++) {
        indexes_layered[i]   = static_cast<IdxT>(indexes_hnsw_host.data_handle()[i]);
        distances_layered[i] = distances_hnsw_host.data_handle()[i];
      }
    }

    // Materialize to a standard hnswlib index file twice: single in-memory pass and bucketed.
    const auto out_single = (std::filesystem::path(temp_dir) / "materialized_single.bin").string();
    const auto out_bucketed =
      (std::filesystem::path(temp_dir) / "materialized_bucketed.bin").string();

    hnsw::materialize_params materialize_single;
    materialize_single.dataset_path       = dataset_file;
    materialize_single.max_host_memory_gb = 0;  // single in-memory reorder pass
    hnsw::materialize_to_hnswlib(
      handle_, materialize_single, artifact_path, out_single, ps.dim, ps.metric);

    hnsw::materialize_params materialize_bucketed;
    materialize_bucketed.dataset_path       = dataset_file;
    materialize_bucketed.max_host_memory_gb = 0.00003;  // tiny budget forces bucketed base + upper
    hnsw::materialize_to_hnswlib(
      handle_, materialize_bucketed, artifact_path, out_bucketed, ps.dim, ps.metric);

    ASSERT_TRUE(std::filesystem::is_regular_file(out_single));
    ASSERT_TRUE(std::filesystem::is_regular_file(out_bucketed));

    // Determinism: single-pass and bucketed outputs must be byte-identical.
    {
      const auto read_all = [](const std::string& path) {
        std::ifstream stream(path, std::ios::binary);
        return std::vector<char>((std::istreambuf_iterator<char>(stream)),
                                 std::istreambuf_iterator<char>());
      };
      const auto bytes_single   = read_all(out_single);
      const auto bytes_bucketed = read_all(out_bucketed);
      ASSERT_FALSE(bytes_single.empty());
      EXPECT_EQ(bytes_single.size(), bytes_bucketed.size());
      EXPECT_TRUE(bytes_single == bytes_bucketed)
        << "Single-pass and bucketed materialized outputs differ";
    }

    // Load the materialized file as a standard (CPU) hnswlib index and search.
    hnsw::index_params cpu_params;
    cpu_params.metric             = ps.metric;
    cpu_params.hierarchy          = hnsw::HnswHierarchy::CPU;
    hnsw::index<DataT>* cpu_index = nullptr;
    hnsw::deserialize(handle_, cpu_params, out_single, ps.dim, ps.metric, &cpu_index);
    ASSERT_NE(cpu_index, nullptr);
    std::unique_ptr<hnsw::index<DataT>> cpu_guard(cpu_index);

    hnsw::search(handle_,
                 search_params,
                 *cpu_guard,
                 queries_host.view(),
                 indexes_hnsw_host.view(),
                 distances_hnsw_host.view());

    std::vector<IdxT> indexes_cpu(queries_size);
    std::vector<DistanceT> distances_cpu(queries_size);
    for (size_t i = 0; i < queries_size; i++) {
      indexes_cpu[i]   = static_cast<IdxT>(indexes_hnsw_host.data_handle()[i]);
      distances_cpu[i] = distances_hnsw_host.data_handle()[i];
    }

    EXPECT_TRUE(cuvs::neighbors::eval_neighbours(indexes_naive,
                                                 indexes_cpu,
                                                 distances_naive,
                                                 distances_cpu,
                                                 ps.n_queries,
                                                 ps.k,
                                                 0.003,
                                                 ps.min_recall))
      << "Materialized hnswlib index failed recall check vs. ground truth";

    // The materialized CPU index represents the same graph and vectors as the layered artifact,
    // so its search results must match the in-memory layered path.
    EXPECT_TRUE(cuvs::neighbors::eval_neighbours(indexes_layered,
                                                 indexes_cpu,
                                                 distances_layered,
                                                 distances_cpu,
                                                 ps.n_queries,
                                                 ps.k,
                                                 0.003,
                                                 0.99))
      << "Materialized hnswlib index disagrees with the in-memory layered path";

    hnsw::materialize_params bad_params;
    bad_params.dataset_path = dataset_file;
    const auto err_out      = (std::filesystem::path(temp_dir) / "err.bin").string();
    EXPECT_THROW(hnsw::materialize_to_hnswlib(
                   handle_, bad_params, artifact_path, err_out, ps.dim + 1, ps.metric),
                 std::exception);

    const auto wrong_metric = ps.metric == cuvs::distance::DistanceType::L2Expanded
                                ? cuvs::distance::DistanceType::InnerProduct
                                : cuvs::distance::DistanceType::L2Expanded;
    EXPECT_THROW(hnsw::materialize_to_hnswlib(
                   handle_, bad_params, artifact_path, err_out, ps.dim, wrong_metric),
                 std::exception);

    hnsw::materialize_params missing_dataset;
    missing_dataset.dataset_path.clear();
    EXPECT_THROW(hnsw::materialize_to_hnswlib(
                   handle_, missing_dataset, artifact_path, err_out, ps.dim, ps.metric),
                 std::exception);
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
  return {
    // Test with L2 metric
    {10,     // n_queries
     5000,   // n_rows
     64,     // dim
     10,     // k
     2,      // npartitions
     100,    // ef_construction
     false,  // use_disk (not explicitly set, should be triggered by memory limit)
     cuvs::distance::DistanceType::L2Expanded,
     0.0,    // min_recall (not checked in fallback test)
     0.001,  // max_host_memory_gb (tiny limit to force disk mode)
     0.001}  // max_gpu_memory_gb (tiny limit to force disk mode)
  };
}

inline std::vector<AnnHnswAceInputs> generate_hnsw_ace_layered_inputs()
{
  return {
    {10,    // n_queries
     5000,  // n_rows
     64,    // dim
     10,    // k
     2,     // npartitions
     100,   // ef_construction
     true,  // use_disk
     cuvs::distance::DistanceType::L2Expanded,
     0.9,  // min_recall
     0.0,  // max_host_memory_gb
     0.0}  // max_gpu_memory_gb
  };
}

const std::vector<AnnHnswAceInputs> hnsw_ace_inputs = generate_hnsw_ace_inputs();
const std::vector<AnnHnswAceInputs> hnsw_ace_memory_fallback_inputs =
  generate_hnsw_ace_memory_fallback_inputs();
const std::vector<AnnHnswAceInputs> hnsw_ace_layered_inputs = generate_hnsw_ace_layered_inputs();

}  // namespace cuvs::neighbors::hnsw
