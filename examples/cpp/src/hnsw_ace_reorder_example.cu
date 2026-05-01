/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// 1. Optionally quantize the dataset and queries to int8.
// 2. Build a CAGRA index using ACE. Store the HNSW layers on disk.
// 3. Build a HNSW index from the layers on disk and search.

// On-disk layout written into `params.index_dir`:
//
//   index_dir/
//     manifest.json
//     cagra_graph_original_ids.npy        # base layer [N, graph_degree]
//     levels.npy                          # uint8  [N]
//     layer_1_points.npy                  # uint32 [n_1]
//     layer_1_graph.npy                   # uint32 [n_1, M]
//     layer_1_degree.npy                  # uint32 [n_1]
//     layer_2_{points,graph,degree}.npy
//     ...

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/random/make_blobs.cuh>

#include <cuvs/neighbors/hnsw.hpp>
#include <cuvs/preprocessing/quantize/scalar.hpp>

#include <rmm/mr/pool_memory_resource.hpp>

#include "common.cuh"

// When 1, scalar-quantize the float dataset to int8.
#define HNSW_ACE_REORDER_USE_QUANTIZATION 1

namespace {

// Single directory holding both ACE scratch artifacts and the layered HNSW
// output. Matches the pattern in `hnsw_ace_example.cu`.
constexpr const char* kBuildDir = "/tmp/hnsw_ace_reorder";

}  // namespace

template <typename T>
struct quantized_pair {
  raft::host_matrix<T, int64_t> dataset;
  raft::host_matrix<T, int64_t> queries;
};

quantized_pair<int8_t> quantize_dataset(raft::device_resources const& dev_resources,
                                        raft::host_matrix_view<const float, int64_t> dataset_float,
                                        raft::host_matrix_view<const float, int64_t> queries_float)
{
  std::cout << "  quantize_dataset: training scalar quantizer (float -> int8)" << std::endl;
  cuvs::preprocessing::quantize::scalar::params qp;
  auto quantizer = cuvs::preprocessing::quantize::scalar::train(dev_resources, qp, dataset_float);

  auto dataset_i8 =
    raft::make_host_matrix<int8_t, int64_t>(dataset_float.extent(0), dataset_float.extent(1));
  cuvs::preprocessing::quantize::scalar::transform(
    dev_resources, quantizer, dataset_float, dataset_i8.view());

  auto queries_i8 =
    raft::make_host_matrix<int8_t, int64_t>(queries_float.extent(0), queries_float.extent(1));
  cuvs::preprocessing::quantize::scalar::transform(
    dev_resources, quantizer, queries_float, queries_i8.view());

  std::cout << "  quantize_dataset: produced int8 dataset [" << dataset_i8.extent(0) << ", "
            << dataset_i8.extent(1) << "] and queries [" << queries_i8.extent(0) << ", "
            << queries_i8.extent(1) << "]" << std::endl;
  return {std::move(dataset_i8), std::move(queries_i8)};
}

template <typename T>
void hnsw_build_to_disk(raft::device_resources const& dev_resources,
                        raft::host_matrix_view<const T, int64_t> dataset,
                        const std::string& build_dir)
{
  using namespace cuvs::neighbors;

  // HNSW index parameters -- identical in spirit to hnsw_ace_example.cu.
  hnsw::index_params hnsw_params;
  hnsw_params.metric    = cuvs::distance::DistanceType::L2Expanded;
  hnsw_params.hierarchy = hnsw::HnswHierarchy::GPU;
  // graph_degree = 2 * M; intermediate_graph_degree = 3 * M. Higher M helps
  // higher-dimensional or high-recall targets at the cost of memory.
  hnsw_params.M = 32;
  // ef_construction controls the candidate list size during hierarchy
  // linking; larger values improve recall with diminishing returns.
  hnsw_params.ef_construction = 120;

  // Parameters for the GPU-accelerated graph builder underneath HNSW.
  auto ace_params = hnsw::graph_build_params::ace_params();
  // Number of ACE partitions. Small values can improve recall at the cost
  // of perf/memory. Watch for imbalance (up to ~3x in practice).
  ace_params.npartitions = 4;
  // Disk-mode ACE writes intermediate artifacts (reordered dataset, dataset
  // mapping, ACE-internal graph) under build_dir. `build_to_disk` reuses
  // this same directory to land the final layered HNSW artifacts, so the
  // whole pipeline produces a single self-describing output directory.
  ace_params.use_disk            = true;
  ace_params.build_dir           = build_dir;
  hnsw_params.graph_build_params = ace_params;

  std::cout << "  hnsw_build_to_disk: writing layered HNSW index to " << build_dir << std::endl;
  hnsw::build_to_disk(dev_resources, hnsw_params, dataset);
  std::cout << "  hnsw_build_to_disk: done (see " << build_dir << "/manifest.json)" << std::endl;
}

template <typename T>
void hnsw_search_from_disk(raft::device_resources const& dev_resources,
                           raft::host_matrix_view<const T, int64_t> dataset,
                           raft::host_matrix_view<const T, int64_t> queries,
                           const std::string& build_dir,
                           int64_t topk = 12)
{
  using namespace cuvs::neighbors;

  std::cout << "  hnsw_search_from_disk: loading layered index from " << build_dir << std::endl;
  // Metric, M, num_layers, entry point, etc. are read from
  // `build_dir/manifest.json`; the dataset argument is what the loaded
  // index is wired against for search-time distance computation.
  auto hnsw_index = hnsw::load_from_disk(dev_resources, build_dir, dataset);

  const int64_t n_queries  = queries.extent(0);
  auto indices_hnsw_host   = raft::make_host_matrix<uint64_t, int64_t>(n_queries, topk);
  auto distances_hnsw_host = raft::make_host_matrix<float, int64_t>(n_queries, topk);

  hnsw::search_params search_params;
  search_params.ef          = std::max(200, static_cast<int>(topk) * 2);
  search_params.num_threads = 1;

  std::cout << "  hnsw_search_from_disk: running HNSW search (top-" << topk << ")" << std::endl;
  hnsw::search(dev_resources,
               search_params,
               *hnsw_index,
               queries,
               indices_hnsw_host.view(),
               distances_hnsw_host.view());

  // Narrow u64 -> u32 for the shared `print_results` helper.
  auto neighbors      = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
  auto distances      = raft::make_device_matrix<float>(dev_resources, n_queries, topk);
  auto neighbors_host = raft::make_host_matrix<uint32_t, int64_t>(n_queries, topk);
  for (int64_t i = 0; i < n_queries; ++i) {
    for (int64_t j = 0; j < topk; ++j) {
      neighbors_host(i, j) = static_cast<uint32_t>(indices_hnsw_host(i, j));
    }
  }
  raft::copy(neighbors.data_handle(),
             neighbors_host.data_handle(),
             n_queries * topk,
             raft::resource::get_cuda_stream(dev_resources));
  raft::copy(distances.data_handle(),
             distances_hnsw_host.data_handle(),
             n_queries * topk,
             raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  print_results(dev_resources, neighbors.view(), distances.view());
}

int main()
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                        1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(pool_mr);

  // Alternatively, one could define a pool allocator for temporary arrays (used within RAFT
  // algorithms). In that case only the internal arrays would use the pool, any other allocation
  // uses the default RMM memory resource. Here is how to change the workspace memory resource to
  // a pool with 2 GiB upper limit.
  // raft::resource::set_workspace_to_pool_resource(dev_resources, 2 * 1024 * 1024 * 1024ull);

#if HNSW_ACE_REORDER_USE_QUANTIZATION
  std::cout << "[stage 1] Generate and quantize dataset (float -> int8)" << std::endl;
#else
  std::cout << "[stage 1] Generate dataset (float)" << std::endl;
#endif

  // ACE requires host-side data, so mirror the generated dataset and queries
  // onto the host.
  auto dataset_host = raft::make_host_matrix<float, int64_t>(n_samples, n_dim);
  raft::copy(dataset_host.data_handle(),
             dataset.data_handle(),
             dataset.extent(0) * dataset.extent(1),
             raft::resource::get_cuda_stream(dev_resources));
  auto queries_host = raft::make_host_matrix<float, int64_t>(n_queries, n_dim);
  raft::copy(queries_host.data_handle(),
             queries.data_handle(),
             queries.extent(0) * queries.extent(1),
             raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  auto dataset_host_view = raft::make_host_matrix_view<const float, int64_t, raft::row_major>(
    dataset_host.data_handle(), n_samples, n_dim);
  auto queries_host_view = raft::make_host_matrix_view<const float, int64_t, raft::row_major>(
    queries_host.data_handle(), n_queries, n_dim);

  std::filesystem::create_directories(kBuildDir);

#if HNSW_ACE_REORDER_USE_QUANTIZATION
  auto q = quantize_dataset(dev_resources, dataset_host_view, queries_host_view);

  auto dataset_i8_view = raft::make_host_matrix_view<const int8_t, int64_t, raft::row_major>(
    q.dataset.data_handle(), n_samples, n_dim);
  auto queries_i8_view = raft::make_host_matrix_view<const int8_t, int64_t, raft::row_major>(
    q.queries.data_handle(), n_queries, n_dim);

  std::cout << "[stage 2] Build HNSW layers and store on disk" << std::endl;
  hnsw_build_to_disk<int8_t>(dev_resources, dataset_i8_view, kBuildDir);

  std::cout << "[stage 3] Build HNSW index from layers on disk and search" << std::endl;
  hnsw_search_from_disk<int8_t>(dev_resources, dataset_i8_view, queries_i8_view, kBuildDir);
#else
  std::cout << "[stage 2] Build HNSW layers and store on disk" << std::endl;
  hnsw_build_to_disk<float>(dev_resources, dataset_host_view, kBuildDir);

  std::cout << "[stage 3] Build HNSW index from layers on disk and search" << std::endl;
  hnsw_search_from_disk<float>(dev_resources, dataset_host_view, queries_host_view, kBuildDir);
#endif

  return 0;
}
