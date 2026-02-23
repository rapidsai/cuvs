/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/graph_build_types.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <utility>

namespace cuvs::neighbors::cagra::helpers {
// Calculate CAGRA optimize workspace memory requirements.
// This is the working memory on top of the input/output memory usage.
std::pair<size_t, size_t> optimize_workspace_size(size_t n_rows,
                                                  size_t graph_degree,
                                                  size_t intermediate_degree,
                                                  size_t index_size,
                                                  bool mst_optimize)
{
  // MST optimization memory (host only)
  size_t mst_host = n_rows * index_size;  // mst_graph_num_edges
  if (mst_optimize) {
    mst_host += n_rows * graph_degree * index_size;  // mst_graph allocated in optimize
    mst_host += n_rows * graph_degree * index_size;  // mst_graph allocated in mst_optimize
    mst_host += n_rows * index_size * 7;             // vectors with _max_edges suffix
    mst_host += (graph_degree - 1) * (graph_degree - 1) * index_size;  // iB_candidates
  }

  // Prune stage memory
  // We neglect 8 bytes (both on host and device) for stats
  size_t prune_host = n_rows * intermediate_degree * sizeof(uint8_t);  // detour count

  size_t prune_dev = n_rows * intermediate_degree * 1;     // detour count (uint8_t)
  prune_dev += n_rows * sizeof(uint32_t);                  // d_num_detour_edges
  prune_dev += n_rows * intermediate_degree * index_size;  // d_input_graph

  // Reverse graph stage memory
  size_t rev_host = n_rows * graph_degree * index_size;  // rev_graph
  rev_host += n_rows * sizeof(uint32_t);                 // rev_graph_count
  rev_host += n_rows * index_size;                       // dest_nodes

  size_t rev_dev = n_rows * graph_degree * index_size;  // d_rev_graph
  rev_dev += n_rows * sizeof(uint32_t);                 // d_rev_graph_count
  rev_dev += n_rows * sizeof(uint32_t);                 // d_dest_nodes

  // Memory for merging graphs (host only)
  size_t combine_host =
    n_rows * sizeof(uint32_t) + graph_degree * sizeof(uint32_t);  // in_edge_count + hist

  size_t total_host = mst_host + std::max({prune_host, rev_host, combine_host});
  size_t total_dev  = std::max(prune_dev, rev_dev);

  return std::make_pair(total_host, total_dev);
}

// All sizes are in bytes
inline std::pair<size_t, size_t> ivf_pq_build_mem_usage(
  raft::resources const& res,
  raft::matrix_extent<int64_t> dataset,
  cuvs::neighbors::graph_build_params::ivf_pq_params params,
  size_t graph_degree,
  size_t intermediate_graph_degree)
{
  size_t n_rows = dataset.extent(0);

  std::cout << "Graph degree " << graph_degree << ", intermediate graph degree "
            << intermediate_graph_degree << std::endl;
  size_t dataset_gpu_mem =
    cuvs::neighbors::ivf_pq::helpers::compressed_dataset_size(res, dataset, params.build_params);
  size_t graph_host_mem = n_rows * (graph_degree + intermediate_graph_degree) * sizeof(uint32_t);

  auto [host_workspace_size, gpu_workspace_size] =
    cuvs::neighbors::cagra::helpers::optimize_workspace_size(
      n_rows, graph_degree, intermediate_graph_degree, sizeof(uint32_t));
  float host_workspace_gb = host_workspace_size / 1e9;
  float gpu_workspace_gb  = gpu_workspace_size / 1e9;

  size_t total_host =
    graph_host_mem + host_workspace_size + 2e9;  // added 2 GB extra workspace (IVF-PQ search)
  size_t total_dev =
    std::max(dataset_gpu_mem, gpu_workspace_size) + 1e9;  // added 1 GB extra workspace size

  std::cout << "IVF-PQ build memory requirements\ndataset_gpu " << dataset_gpu_mem / 1e9 << " GB"
            << std::endl;
  std::cout << "graph_host_mem " << graph_host_mem / 1e9 << " GB" << ", workspace "
            << host_workspace_gb << " GB" << std::endl;
  std::cout << "gpu mem" << gpu_workspace_gb << " GB" << std::endl;
  return std::make_pair(total_host, total_dev);
}

std::pair<size_t, size_t> cagra_build_mem_usage(raft::resources const& res,
                                                raft::matrix_extent<int64_t> dataset,
                                                size_t dtype_size,
                                                cuvs::neighbors::cagra::index_params cparams)
{
  using namespace cuvs::neighbors;

  size_t total_host = 0;
  size_t total_dev  = 0;

  if (std::holds_alternative<graph_build_params::ivf_pq_params>(cparams.graph_build_params)) {
    RAFT_LOG_INFO("Considering CAGRA in memory build with IVF-PQ");
    graph_build_params::ivf_pq_params pq_params =
      std::get<graph_build_params::ivf_pq_params>(cparams.graph_build_params);
    std::tie(total_host, total_dev) = ivf_pq_build_mem_usage(
      res, dataset, pq_params, cparams.graph_degree, cparams.intermediate_graph_degree);
  } else if (std::holds_alternative<graph_build_params::nn_descent_params>(
               cparams.graph_build_params)) {
    RAFT_LOG_INFO("Considering CAGRA in memory build with NN-descent");
    // Build needs dataset in fp16 on dev and graph on dev?
    // dataset copied to device for sorting
    // TODO(tfeher) proper estimate
    total_host = dataset.extent(0) * dataset.extent(1) * dtype_size +
                 dataset.extent(0) * (cparams.graph_degree + cparams.intermediate_graph_degree) *
                   sizeof(uint32_t) +
                 2e9;  // Extra buffer
    total_dev = total_host;
  } else {
    // iterative build
    // TODO(tfeher): proper estimate
    total_host = dataset.extent(0) * dataset.extent(1) * dtype_size +
                 dataset.extent(0) * (cparams.graph_degree + cparams.intermediate_graph_degree) *
                   sizeof(uint32_t) +
                 2e9;  // Extra buffer
    total_dev = total_host;
  }
  return std::make_pair(total_host, total_dev);
}

}  // namespace cuvs::neighbors::cagra::helpers
