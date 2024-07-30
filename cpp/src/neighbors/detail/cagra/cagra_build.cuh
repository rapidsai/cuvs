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

#include "../../vpq_dataset.cuh"
#include "graph_core.cuh"
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger-ext.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/refine.hpp>

// TODO: Fixme- this needs to be migrated
#include "../../ivf_pq/ivf_pq_build.cuh"
#include "../../ivf_pq/ivf_pq_search.cuh"
#include "../../nn_descent.cuh"

// TODO: This shouldn't be calling spatial/knn APIs
#include "../ann_utils.cuh"

#include <rmm/resource_ref.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace cuvs::neighbors::cagra::detail {

static const std::string RAFT_NAME = "raft";

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
  raft::common::nvtx::range<raft::common::nvtx::domain::raft> fun_scope(
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
  auto index =
    cuvs::neighbors::ivf_pq::detail::build<DataT, int64_t>(res, pq.build_params, dataset);

  //
  // search top (k + 1) neighbors
  //

  const auto top_k       = node_degree + 1;
  uint32_t gpu_top_k     = node_degree * pq.refinement_rate;
  gpu_top_k              = std::min<IdxT>(std::max(gpu_top_k, top_k), dataset.extent(0));
  const auto num_queries = dataset.extent(0);

  // Use the same maximum batch size as the ivf_pq::search to avoid allocating more than needed.
  using cuvs::neighbors::ivf_pq::detail::kMaxQueries;
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
  auto nn_descent_idx = cuvs::neighbors::nn_descent::index<IdxT>(res, knn_graph);
  cuvs::neighbors::nn_descent::build<DataT, IdxT>(res, build_params, dataset, nn_descent_idx);

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

  cuvs::neighbors::cagra::detail::graph::sort_knn_graph(res, dataset, knn_graph_internal);
}

template <
  typename IdxT = uint32_t,
  typename g_accessor =
    raft::host_device_accessor<std::experimental::default_accessor<IdxT>, raft::memory_type::host>>
void optimize(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor> knn_graph,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> new_graph)
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

  cagra::detail::graph::optimize(res, knn_graph_internal, new_graph_internal);
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

  std::optional<raft::host_matrix<IdxT, int64_t>> knn_graph(
    raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), intermediate_degree));

  // Set default value in case knn_build_params is not defined.
  auto knn_build_params = params.graph_build_params;
  if (std::holds_alternative<std::monostate>(params.graph_build_params)) {
    // Heuristic to decide default build algo and its params.
    if (params.metric == cuvs::distance::DistanceType::L2Expanded &&
        cuvs::neighbors::nn_descent::has_enough_device_memory(
          res, dataset.extents(), sizeof(IdxT))) {
      RAFT_LOG_DEBUG("NN descent solver");
      knn_build_params = cagra::graph_build_params::nn_descent_params(intermediate_degree);
    } else {
      RAFT_LOG_DEBUG("Selecting IVF-PQ solver");
      knn_build_params = cagra::graph_build_params::ivf_pq_params(dataset.extents(), params.metric);
    }
  }

  // Dispatch based on graph_build_params
  if (std::holds_alternative<cagra::graph_build_params::ivf_pq_params>(knn_build_params)) {
    auto ivf_pq_params =
      std::get<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(knn_build_params);
    build_knn_graph(res, dataset, knn_graph->view(), ivf_pq_params);
  } else {
    RAFT_EXPECTS(
      params.metric == cuvs::distance::DistanceType::L2Expanded,
      "L2Expanded is the only distance metrics supported for CAGRA build with nn_descent");
    auto nn_descent_params =
      std::get<cagra::graph_build_params::nn_descent_params>(knn_build_params);

    if (nn_descent_params.graph_degree != intermediate_degree) {
      RAFT_LOG_WARN(
        "Graph degree (%lu) for nn-descent needs to match cagra intermediate graph degree (%lu), "
        "aligning "
        "nn-descent graph_degree.",
        nn_descent_params.graph_degree,
        intermediate_degree);
      nn_descent_params = cagra::graph_build_params::nn_descent_params(intermediate_degree);
    }

    // Use nn-descent to build CAGRA knn graph
    build_knn_graph<T, IdxT>(res, dataset, knn_graph->view(), nn_descent_params);
  }

  auto cagra_graph = raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), graph_degree);

  RAFT_LOG_INFO("optimizing graph");
  optimize<IdxT>(res, knn_graph->view(), cagra_graph.view());

  // free intermediate graph before trying to create the index
  knn_graph.reset();

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
