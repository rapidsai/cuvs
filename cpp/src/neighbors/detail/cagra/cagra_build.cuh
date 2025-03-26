/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "../../../core/nvtx.hpp"
#include "../../vpq_dataset.cuh"
#include "graph_core.cuh"
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/refine.hpp>

#include <cuvs/neighbors/nn_descent.hpp>

// TODO: This shouldn't be calling spatial/knn APIs
#include "../ann_utils.cuh"

#include <rmm/resource_ref.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace cuvs::neighbors::cagra::detail {

template <typename IdxT>
void write_to_graph(raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
                    raft::host_matrix_view<int64_t, int64_t, raft::row_major> neighbors_host_view,
                    size_t& num_self_included,
                    size_t batch_size,
                    size_t batch_offset)
{
  uint32_t node_degree = knn_graph.extent(1);
  size_t top_k         = neighbors_host_view.extent(1);
  // omit itself & write out
  for (std::size_t i = 0; i < batch_size; i++) {
    size_t vec_idx = i + batch_offset;
    for (std::size_t j = 0, num_added = 0; j < top_k && num_added < node_degree; j++) {
      const auto v = neighbors_host_view(i, j);
      if (static_cast<size_t>(v) == vec_idx) {
        num_self_included++;
        continue;
      }
      knn_graph(vec_idx, num_added) = v;
      num_added++;
    }
  }
}

template <typename DataT, typename IdxT, typename accessor>
void refine_host_and_write_graph(
  raft::resources const& res,
  raft::host_matrix<DataT, int64_t>& queries_host,
  raft::host_matrix<int64_t, int64_t>& neighbors_host,
  raft::host_matrix<int64_t, int64_t>& refined_neighbors_host,
  raft::host_matrix<float, int64_t>& refined_distances_host,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
  cuvs::distance::DistanceType metric,
  size_t& num_self_included,
  size_t batch_size,
  size_t batch_offset,
  int top_k,
  int gpu_top_k)
{
  bool do_refine = top_k != gpu_top_k;

  auto refined_neighbors_host_view = raft::make_host_matrix_view<int64_t, int64_t>(
    do_refine ? refined_neighbors_host.data_handle() : neighbors_host.data_handle(),
    batch_size,
    top_k);

  if (do_refine) {
    // needed for compilation as this routine will also be run for device data with !do_refine
    if constexpr (raft::is_host_mdspan_v<decltype(dataset)>) {
      auto queries_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
        queries_host.data_handle(), batch_size, dataset.extent(1));
      auto neighbors_host_view = raft::make_host_matrix_view<const int64_t, int64_t>(
        neighbors_host.data_handle(), batch_size, neighbors_host.extent(1));
      auto refined_distances_host_view = raft::make_host_matrix_view<float, int64_t>(
        refined_distances_host.data_handle(), batch_size, top_k);
      cuvs::neighbors::refine(res,
                              dataset,
                              queries_host_view,
                              neighbors_host_view,
                              refined_neighbors_host_view,
                              refined_distances_host_view,
                              metric);
    }
  }

  write_to_graph(
    knn_graph, refined_neighbors_host_view, num_self_included, batch_size, batch_offset);
}

template <typename DataT, typename IdxT, typename accessor>
void build_knn_graph(
  raft::resources const& res,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
  cuvs::neighbors::cagra::graph_build_params::ivf_pq_params pq)
{
  RAFT_EXPECTS(pq.build_params.metric == cuvs::distance::DistanceType::L2Expanded ||
                 pq.build_params.metric == cuvs::distance::DistanceType::InnerProduct,
               "Currently only L2Expanded or InnerProduct metric are supported");

  uint32_t node_degree = knn_graph.extent(1);
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "cagra::build_graph(%zu, %zu, %u)",
    size_t(dataset.extent(0)),
    size_t(dataset.extent(1)),
    node_degree);

  // Make model name
  const std::string model_name = [&]() {
    char model_name[1024];
    sprintf(model_name,
            "%s-%lux%lu.cluster_%u.pq_%u.%ubit.itr_%u.metric_%u.pqcenter_%u",
            "IVF-PQ",
            static_cast<size_t>(dataset.extent(0)),
            static_cast<size_t>(dataset.extent(1)),
            pq.build_params.n_lists,
            pq.build_params.pq_dim,
            pq.build_params.pq_bits,
            pq.build_params.kmeans_n_iters,
            pq.build_params.metric,
            static_cast<uint32_t>(pq.build_params.codebook_kind));
    return std::string(model_name);
  }();

  RAFT_LOG_DEBUG("# Building IVF-PQ index %s", model_name.c_str());
  auto index = cuvs::neighbors::ivf_pq::build(res, pq.build_params, dataset);

  //
  // search top (k + 1) neighbors
  //

  const auto top_k       = node_degree + 1;
  uint32_t gpu_top_k     = node_degree * pq.refinement_rate;
  gpu_top_k              = std::min<IdxT>(std::max(gpu_top_k, top_k), dataset.extent(0));
  const auto num_queries = dataset.extent(0);

  // Use the same maximum batch size as the ivf_pq::search to avoid allocating more than needed.
  constexpr uint32_t kMaxQueries = 4096;

  // Heuristic: the build_knn_graph code should use only a fraction of the workspace memory; the
  // rest should be used by the ivf_pq::search. Here we say that the workspace size should be a good
  // multiple of what is required for the I/O batching below.
  constexpr size_t kMinWorkspaceRatio = 5;
  auto desired_workspace_size         = kMaxQueries * kMinWorkspaceRatio *
                                (sizeof(DataT) * dataset.extent(1)  // queries (dataset batch)
                                 + sizeof(float) * gpu_top_k        // distances
                                 + sizeof(int64_t) * gpu_top_k      // neighbors
                                 + sizeof(float) * top_k            // refined_distances
                                 + sizeof(int64_t) * top_k          // refined_neighbors
                                );

  // If the workspace is smaller than desired, put the I/O buffers into the large workspace.
  rmm::device_async_resource_ref workspace_mr =
    desired_workspace_size <= raft::resource::get_workspace_free_bytes(res)
      ? raft::resource::get_workspace_resource(res)
      : raft::resource::get_large_workspace_resource(res);

  RAFT_LOG_DEBUG(
    "IVF-PQ search node_degree: %d, top_k: %d,  gpu_top_k: %d,  max_batch_size:: %d, n_probes: %u",
    node_degree,
    top_k,
    gpu_top_k,
    kMaxQueries,
    pq.search_params.n_probes);

  auto distances = raft::make_device_mdarray<float>(
    res, workspace_mr, raft::make_extents<int64_t>(kMaxQueries, gpu_top_k));
  auto neighbors = raft::make_device_mdarray<int64_t>(
    res, workspace_mr, raft::make_extents<int64_t>(kMaxQueries, gpu_top_k));
  auto refined_distances = raft::make_device_mdarray<float>(
    res, workspace_mr, raft::make_extents<int64_t>(kMaxQueries, top_k));
  auto refined_neighbors = raft::make_device_mdarray<int64_t>(
    res, workspace_mr, raft::make_extents<int64_t>(kMaxQueries, top_k));
  auto neighbors_host = raft::make_host_matrix<int64_t, int64_t>(kMaxQueries, gpu_top_k);
  auto queries_host   = raft::make_host_matrix<DataT, int64_t>(kMaxQueries, dataset.extent(1));
  auto refined_neighbors_host = raft::make_host_matrix<int64_t, int64_t>(kMaxQueries, top_k);
  auto refined_distances_host = raft::make_host_matrix<float, int64_t>(kMaxQueries, top_k);

  // TODO(tfeher): batched search with multiple GPUs
  std::size_t num_self_included = 0;
  bool first                    = true;
  const auto start_clock        = std::chrono::system_clock::now();

  cuvs::spatial::knn::detail::utils::batch_load_iterator<DataT> vec_batches(
    dataset.data_handle(),
    dataset.extent(0),
    dataset.extent(1),
    static_cast<int64_t>(kMaxQueries),
    raft::resource::get_cuda_stream(res),
    workspace_mr);

  size_t next_report_offset = 0;
  size_t d_report_offset    = dataset.extent(0) / 100;  // Report progress in 1% steps.

  bool async_host_processing   = raft::is_host_mdspan_v<decltype(dataset)> || top_k == gpu_top_k;
  size_t previous_batch_size   = 0;
  size_t previous_batch_offset = 0;

  for (const auto& batch : vec_batches) {
    // Map int64_t to uint32_t because ivf_pq requires the latter.
    // TODO(tfeher): remove this mapping once ivf_pq accepts mdspan with int64_t index type
    auto queries_view = raft::make_device_matrix_view<const DataT, uint32_t>(
      batch.data(), batch.size(), batch.row_width());
    auto neighbors_view = raft::make_device_matrix_view<int64_t, uint32_t>(
      neighbors.data_handle(), batch.size(), neighbors.extent(1));
    auto distances_view = raft::make_device_matrix_view<float, uint32_t>(
      distances.data_handle(), batch.size(), distances.extent(1));

    cuvs::neighbors::ivf_pq::search(
      res, pq.search_params, index, queries_view, neighbors_view, distances_view);

    if (async_host_processing) {
      // process previous batch async on host
      // NOTE: the async path also covers disabled refinement (top_k == gpu_top_k)
      if (previous_batch_size > 0) {
        refine_host_and_write_graph(res,
                                    queries_host,
                                    neighbors_host,
                                    refined_neighbors_host,
                                    refined_distances_host,
                                    dataset,
                                    knn_graph,
                                    pq.build_params.metric,
                                    num_self_included,
                                    previous_batch_size,
                                    previous_batch_offset,
                                    top_k,
                                    gpu_top_k);
      }

      // copy next batch to host
      raft::copy(neighbors_host.data_handle(),
                 neighbors.data_handle(),
                 neighbors_view.size(),
                 raft::resource::get_cuda_stream(res));
      if (top_k != gpu_top_k) {
        // can be skipped for disabled refinement
        raft::copy(queries_host.data_handle(),
                   batch.data(),
                   queries_view.size(),
                   raft::resource::get_cuda_stream(res));
      }

      previous_batch_size   = batch.size();
      previous_batch_offset = batch.offset();

      // we need to ensure the copy operations are done prior using the host data
      raft::resource::sync_stream(res);

      // process last batch
      if (previous_batch_offset + previous_batch_size == (size_t)num_queries) {
        refine_host_and_write_graph(res,
                                    queries_host,
                                    neighbors_host,
                                    refined_neighbors_host,
                                    refined_distances_host,
                                    dataset,
                                    knn_graph,
                                    pq.build_params.metric,
                                    num_self_included,
                                    previous_batch_size,
                                    previous_batch_offset,
                                    top_k,
                                    gpu_top_k);
      }
    } else {
      auto neighbor_candidates_view = raft::make_device_matrix_view<const int64_t, uint64_t>(
        neighbors.data_handle(), batch.size(), gpu_top_k);
      auto refined_neighbors_view = raft::make_device_matrix_view<int64_t, int64_t>(
        refined_neighbors.data_handle(), batch.size(), top_k);
      auto refined_distances_view = raft::make_device_matrix_view<float, int64_t>(
        refined_distances.data_handle(), batch.size(), top_k);

      auto dataset_view = raft::make_device_matrix_view<const DataT, int64_t>(
        dataset.data_handle(), dataset.extent(0), dataset.extent(1));
      cuvs::neighbors::refine(res,
                              dataset_view,
                              queries_view,
                              neighbor_candidates_view,
                              refined_neighbors_view,
                              refined_distances_view,
                              pq.build_params.metric);
      raft::copy(refined_neighbors_host.data_handle(),
                 refined_neighbors_view.data_handle(),
                 refined_neighbors_view.size(),
                 raft::resource::get_cuda_stream(res));
      raft::resource::sync_stream(res);

      auto refined_neighbors_host_view = raft::make_host_matrix_view<int64_t, int64_t>(
        refined_neighbors_host.data_handle(), batch.size(), top_k);
      write_to_graph(
        knn_graph, refined_neighbors_host_view, num_self_included, batch.size(), batch.offset());
    }

    size_t num_queries_done = batch.offset() + batch.size();
    const auto end_clock    = std::chrono::system_clock::now();
    if (batch.offset() > next_report_offset) {
      next_report_offset += d_report_offset;
      const auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() *
        1e-6;
      const auto throughput = num_queries_done / time;

      RAFT_LOG_DEBUG(
        "# Search %12lu / %12lu (%3.2f %%), %e queries/sec, %.2f minutes ETA, self included = "
        "%3.2f %%    \r",
        num_queries_done,
        dataset.extent(0),
        num_queries_done / static_cast<double>(dataset.extent(0)) * 100,
        throughput,
        (num_queries - num_queries_done) / throughput / 60,
        static_cast<double>(num_self_included) / num_queries_done * 100.);
    }
    first = false;
  }

  if (!first) RAFT_LOG_DEBUG("# Finished building kNN graph");
}

template <typename DataT, typename IdxT, typename accessor>
void build_knn_graph(
  raft::resources const& res,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
  cuvs::neighbors::nn_descent::index_params build_params)
{
  std::optional<raft::host_matrix_view<IdxT, int64_t, row_major>> graph_view = knn_graph;
  auto nn_descent_idx = cuvs::neighbors::nn_descent::build(res, build_params, dataset, graph_view);

  using internal_IdxT = typename std::make_unsigned<IdxT>::type;
  using g_accessor    = typename decltype(nn_descent_idx.graph())::accessor_type;
  using g_accessor_internal =
    raft::host_device_accessor<std::experimental::default_accessor<internal_IdxT>,
                               g_accessor::mem_type>;

  auto knn_graph_internal =
    raft::mdspan<internal_IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor_internal>(
      reinterpret_cast<internal_IdxT*>(nn_descent_idx.graph().data_handle()),
      nn_descent_idx.graph().extent(0),
      nn_descent_idx.graph().extent(1));

  cuvs::neighbors::cagra::detail::graph::sort_knn_graph(
    res, build_params.metric, dataset, knn_graph_internal);
}

template <
  typename IdxT = uint32_t,
  typename g_accessor =
    raft::host_device_accessor<std::experimental::default_accessor<IdxT>, raft::memory_type::host>>
void optimize(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor> knn_graph,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> new_graph,
  const bool guarantee_connectivity = false)
{
  using internal_IdxT = typename std::make_unsigned<IdxT>::type;

  auto new_graph_internal = raft::make_host_matrix_view<internal_IdxT, int64_t>(
    reinterpret_cast<internal_IdxT*>(new_graph.data_handle()),
    new_graph.extent(0),
    new_graph.extent(1));

  using g_accessor_internal =
    raft::host_device_accessor<std::experimental::default_accessor<internal_IdxT>,
                               raft::memory_type::host>;
  auto knn_graph_internal =
    raft::mdspan<internal_IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor_internal>(
      reinterpret_cast<internal_IdxT*>(knn_graph.data_handle()),
      knn_graph.extent(0),
      knn_graph.extent(1));

  cagra::detail::graph::optimize(
    res, knn_graph_internal, new_graph_internal, guarantee_connectivity);
}

template <typename T,
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
auto iterative_build_graph(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;

  auto cagra_graph = raft::make_host_matrix<IdxT, int64_t>(0, 0);

  // Iteratively improve the accuracy of the graph by repeatedly running
  // CAGRA's search() and optimize(). As for the size of the graph, instead
  // of targeting all nodes from the beginning, the number of nodes is
  // initially small, and the number of nodes is doubled with each iteration.
  RAFT_LOG_INFO("Iteratively creating/improving graph index using CAGRA's search() and optimize()");

  // If dataset is a host matrix, change it to a device matrix. Also, if the
  // dimensionality of the dataset does not meet the alighnemt restriction,
  // add extra dimensions and change it to a strided matrix.
  std::unique_ptr<strided_dataset<T, int64_t>> dev_aligned_dataset;
  try {
    dev_aligned_dataset = make_aligned_dataset(res, dataset);
  } catch (raft::logic_error& e) {
    RAFT_LOG_ERROR("Iterative CAGRA graph build requires the dataset to fit GPU memory");
    throw e;
  }
  auto dev_aligned_dataset_view = dev_aligned_dataset.get()->view();

  // If the matrix stride and extent do no match, the extra dimensions are
  // also as extent since it cannot be used as query matrix.
  auto dev_dataset =
    raft::make_device_matrix_view<const T, int64_t>(dev_aligned_dataset_view.data_handle(),
                                                    dev_aligned_dataset_view.extent(0),
                                                    dev_aligned_dataset_view.stride(0));

  // Determine initial graph size.
  uint64_t final_graph_size   = (uint64_t)dataset.extent(0);
  uint64_t initial_graph_size = (final_graph_size + 1) / 2;
  while (initial_graph_size > graph_degree * 64) {
    initial_graph_size = (initial_graph_size + 1) / 2;
  }
  RAFT_LOG_DEBUG("# initial graph size = %lu", (uint64_t)initial_graph_size);

  // Allocate memory for search results.
  constexpr uint64_t max_chunk_size = 8192;
  auto topk                         = intermediate_degree;
  auto dev_neighbors = raft::make_device_matrix<IdxT, int64_t>(res, max_chunk_size, topk);
  auto dev_distances = raft::make_device_matrix<float, int64_t>(res, max_chunk_size, topk);

  // Determine graph degree and number of search results while increasing
  // graph size.
  auto small_graph_degree = std::max(graph_degree / 2, std::min(graph_degree, (uint64_t)32));
  auto small_topk         = topk * small_graph_degree / graph_degree;
  RAFT_LOG_DEBUG("# graph_degree = %lu", (uint64_t)graph_degree);
  RAFT_LOG_DEBUG("# small_graph_degree = %lu", (uint64_t)small_graph_degree);
  RAFT_LOG_DEBUG("# topk = %lu", (uint64_t)topk);
  RAFT_LOG_DEBUG("# small_topk = %lu", (uint64_t)small_topk);

  // Create an initial graph. The initial graph created here is not suitable for
  // searching, but connectivity is guaranteed.
  auto offset       = raft::make_host_vector<IdxT, int64_t>(small_graph_degree);
  const double base = sqrt((double)2.0);
  for (uint64_t j = 0; j < small_graph_degree; j++) {
    if (j == 0) {
      offset(j) = 1;
    } else {
      offset(j) = offset(j - 1) + 1;
    }
    IdxT ofst = initial_graph_size * pow(base, (double)j - small_graph_degree - 1);
    if (offset(j) < ofst) { offset(j) = ofst; }
    RAFT_LOG_DEBUG("# offset(%lu) = %lu\n", (uint64_t)j, (uint64_t)offset(j));
  }
  cagra_graph = raft::make_host_matrix<IdxT, int64_t>(initial_graph_size, small_graph_degree);
  for (uint64_t i = 0; i < initial_graph_size; i++) {
    for (uint64_t j = 0; j < small_graph_degree; j++) {
      cagra_graph(i, j) = (i + offset(j)) % initial_graph_size;
    }
  }

  auto curr_graph_size = initial_graph_size;
  while (true) {
    RAFT_LOG_DEBUG("# graph_size = %lu (%.3lf)",
                   (uint64_t)curr_graph_size,
                   (double)curr_graph_size / final_graph_size);

    auto curr_query_size   = std::min(2 * curr_graph_size, final_graph_size);
    auto curr_topk         = small_topk;
    auto curr_itopk_size   = small_topk * 3 / 2;
    auto curr_graph_degree = small_graph_degree;
    if (curr_query_size == final_graph_size) {
      curr_topk         = topk;
      curr_itopk_size   = topk * 2;
      curr_graph_degree = graph_degree;
    }

    cuvs::neighbors::cagra::search_params search_params;
    search_params.algo        = cuvs::neighbors::cagra::search_algo::AUTO;
    search_params.max_queries = max_chunk_size;
    search_params.itopk_size  = curr_itopk_size;

    // Create an index (idx), a query view (dev_query_view), and a mdarray for
    // search results (neighbors).
    auto dev_dataset_view = raft::make_device_matrix_view<const T, int64_t>(
      dev_dataset.data_handle(), (int64_t)curr_graph_size, dev_dataset.extent(1));

    auto idx = index<T, IdxT>(
      res, params.metric, dev_dataset_view, raft::make_const_mdspan(cagra_graph.view()));

    auto dev_query_view = raft::make_device_matrix_view<const T, int64_t>(
      dev_dataset.data_handle(), (int64_t)curr_query_size, dev_dataset.extent(1));
    auto neighbors = raft::make_host_matrix<IdxT, int64_t>(curr_query_size, curr_topk);

    // Search.
    // Since there are many queries, divide them into batches and search them.
    cuvs::spatial::knn::detail::utils::batch_load_iterator<T> query_batch(
      dev_query_view.data_handle(),
      curr_query_size,
      dev_query_view.extent(1),
      max_chunk_size,
      raft::resource::get_cuda_stream(res),
      raft::resource::get_workspace_resource(res));
    for (const auto& batch : query_batch) {
      auto batch_dev_query_view = raft::make_device_matrix_view<const T, int64_t>(
        batch.data(), batch.size(), dev_query_view.extent(1));
      auto batch_dev_neighbors_view = raft::make_device_matrix_view<IdxT, int64_t>(
        dev_neighbors.data_handle(), batch.size(), curr_topk);
      auto batch_dev_distances_view = raft::make_device_matrix_view<float, int64_t>(
        dev_distances.data_handle(), batch.size(), curr_topk);

      cuvs::neighbors::cagra::search(res,
                                     search_params,
                                     idx,
                                     batch_dev_query_view,
                                     batch_dev_neighbors_view,
                                     batch_dev_distances_view);

      auto batch_neighbors_view = raft::make_host_matrix_view<IdxT, int64_t>(
        neighbors.data_handle() + batch.offset() * curr_topk, batch.size(), curr_topk);
      raft::copy(batch_neighbors_view.data_handle(),
                 batch_dev_neighbors_view.data_handle(),
                 batch_neighbors_view.size(),
                 raft::resource::get_cuda_stream(res));
    }

    // Optimize graph
    bool flag_last  = (curr_graph_size == final_graph_size);
    curr_graph_size = curr_query_size;
    cagra_graph     = raft::make_host_matrix<IdxT, int64_t>(0, 0);  // delete existing grahp
    cagra_graph     = raft::make_host_matrix<IdxT, int64_t>(curr_graph_size, curr_graph_degree);
    optimize<IdxT>(
      res, neighbors.view(), cagra_graph.view(), flag_last ? params.guarantee_connectivity : 0);
    if (flag_last) { break; }
  }

  return cagra_graph;
}

template <typename T,
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
index<T, IdxT> build(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;
  if (intermediate_degree >= static_cast<size_t>(dataset.extent(0))) {
    RAFT_LOG_WARN(
      "Intermediate graph degree cannot be larger than dataset size, reducing it to %lu",
      dataset.extent(0));
    intermediate_degree = dataset.extent(0) - 1;
  }
  if (intermediate_degree < graph_degree) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree,
      intermediate_degree);
    graph_degree = intermediate_degree;
  }

  // Set default value in case knn_build_params is not defined.
  auto knn_build_params = params.graph_build_params;
  if (std::holds_alternative<std::monostate>(params.graph_build_params)) {
    // Heuristic to decide default build algo and its params.
    if (cuvs::neighbors::nn_descent::has_enough_device_memory(
          res, dataset.extents(), sizeof(IdxT))) {
      RAFT_LOG_DEBUG("NN descent solver");
      knn_build_params =
        cagra::graph_build_params::nn_descent_params(intermediate_degree, params.metric);
    } else {
      RAFT_LOG_DEBUG("Selecting IVF-PQ solver");
      knn_build_params = cagra::graph_build_params::ivf_pq_params(dataset.extents(), params.metric);
    }
  }
  RAFT_EXPECTS(
    params.metric != BitwiseHamming ||
      std::holds_alternative<cagra::graph_build_params::iterative_search_params>(knn_build_params),
    "IVF_PQ and NN_DESCENT for CAGRA graph build do not support BitwiseHamming as a metric. Please "
    "use the iterative CAGRA search build.");

  auto cagra_graph = raft::make_host_matrix<IdxT, int64_t>(0, 0);

  // Dispatch based on graph_build_params
  if (std::holds_alternative<cagra::graph_build_params::iterative_search_params>(
        knn_build_params)) {
    cagra_graph = iterative_build_graph<T, IdxT, Accessor>(res, params, dataset);
  } else {
    std::optional<raft::host_matrix<IdxT, int64_t>> knn_graph(
      raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), intermediate_degree));

    if (std::holds_alternative<cagra::graph_build_params::ivf_pq_params>(knn_build_params)) {
      auto ivf_pq_params =
        std::get<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(knn_build_params);
      build_knn_graph(res, dataset, knn_graph->view(), ivf_pq_params);
    } else {
      auto nn_descent_params =
        std::get<cagra::graph_build_params::nn_descent_params>(knn_build_params);

      if (nn_descent_params.graph_degree != intermediate_degree) {
        RAFT_LOG_WARN(
          "Graph degree (%lu) for nn-descent needs to match cagra intermediate graph degree (%lu), "
          "aligning "
          "nn-descent graph_degree.",
          nn_descent_params.graph_degree,
          intermediate_degree);
        nn_descent_params =
          cagra::graph_build_params::nn_descent_params(intermediate_degree, params.metric);
      }

      // Use nn-descent to build CAGRA knn graph
      nn_descent_params.return_distances = false;
      build_knn_graph<T, IdxT>(res, dataset, knn_graph->view(), nn_descent_params);
    }

    cagra_graph = raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), graph_degree);

    RAFT_LOG_INFO("optimizing graph");
    optimize<IdxT>(res, knn_graph->view(), cagra_graph.view(), params.guarantee_connectivity);

    // free intermediate graph before trying to create the index
    knn_graph.reset();
  }

  RAFT_LOG_INFO("Graph optimized, creating index");

  // Construct an index from dataset and optimized knn graph.
  if (params.compression.has_value()) {
    RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded,
                 "VPQ compression is only supported with L2Expanded distance mertric");
    index<T, IdxT> idx(res, params.metric);
    idx.update_graph(res, raft::make_const_mdspan(cagra_graph.view()));
    idx.update_dataset(
      res,
      // TODO: hardcoding codebook math to `half`, we can do runtime dispatching later
      cuvs::neighbors::vpq_build<decltype(dataset), half, int64_t>(
        res, *params.compression, dataset));

    return idx;
  }
  if (params.attach_dataset_on_build) {
    try {
      return index<T, IdxT>(
        res, params.metric, dataset, raft::make_const_mdspan(cagra_graph.view()));
    } catch (std::bad_alloc& e) {
      RAFT_LOG_WARN(
        "Insufficient GPU memory to construct CAGRA index with dataset on GPU. Only the graph will "
        "be added to the index");
      // We just add the graph. User is expected to update dataset separately (e.g allocating in
      // managed memory).
    } catch (raft::logic_error& e) {
      // The memory error can also manifest as logic_error.
      RAFT_LOG_WARN(
        "Insufficient GPU memory to construct CAGRA index with dataset on GPU. Only the graph will "
        "be added to the index");
    }
  }
  index<T, IdxT> idx(res, params.metric);
  idx.update_graph(res, raft::make_const_mdspan(cagra_graph.view()));
  return idx;
}
}  // namespace cuvs::neighbors::cagra::detail
