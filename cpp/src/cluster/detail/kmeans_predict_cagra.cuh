/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Cluster assignment via CAGRA: assign each data point to nearest centroid using an
 * approximate nearest neighbor search (CAGRA) over the centroids instead of brute force.
 * Used for scaling IVF training when the number of clusters K is very large.
 */
#pragma once

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <type_traits>

namespace cuvs::cluster::kmeans::detail {

/**
 * @brief Assign each row in X to the nearest centroid using CAGRA (1-NN search over centroids).
 *
 * Builds a CAGRA index on the centroids and runs k=1 search with X as queries. The returned
 * neighbor indices are the cluster labels. This is approximate and faster than brute force
 * when the number of clusters K is large.
 *
 * Supports the same metrics as CAGRA: L2Expanded, L2SqrtExpanded, InnerProduct, CosineExpanded.
 * Centroids and (after mapping) query data must be float.
 *
 * @param[in]  handle    RAFT resources
 * @param[in]  params    Balanced params (metric used for assignment)
 * @param[in]  X         Data to assign [n_rows, dim]
 * @param[in]  centroids Cluster centers [n_clusters, dim] (device, row-major)
 * @param[out] labels    Output cluster index per row [n_rows]
 * @param[in]  mapping_op Optional mapping from DataT to float (e.g. for quantized input)
 */
template <typename DataT,
          typename MathT,
          typename IdxT,
          typename LabelT,
          typename MappingOpT = raft::identity_op>
std::enable_if_t<std::is_same_v<MathT, float>> predict_cagra(
  raft::resources const& handle,
  cuvs::cluster::kmeans::balanced_params const& params,
  raft::device_matrix_view<const DataT, IdxT> X,
  raft::device_matrix_view<const MathT, IdxT> centroids,
  raft::device_vector_view<LabelT, IdxT> labels,
  MappingOpT mapping_op = raft::identity_op())
{
  using namespace cuvs::neighbors::cagra;

  RAFT_EXPECTS(X.extent(0) == labels.extent(0), "X rows and labels size must match");
  RAFT_EXPECTS(X.extent(1) == centroids.extent(1), "X dim and centroids dim must match");
  RAFT_EXPECTS(centroids.extent(0) >= 1, "Need at least one centroid");

  auto stream = raft::resource::get_cuda_stream(handle);
  int64_t n_rows     = static_cast<int64_t>(X.extent(0));
  int64_t n_clusters = static_cast<int64_t>(centroids.extent(0));
  int64_t dim        = static_cast<int64_t>(centroids.extent(1));

  // CAGRA graph degree cannot exceed n_clusters - 1
  size_t graph_degree = std::min<size_t>(64, std::max<int64_t>(1, n_clusters - 1));
  size_t inter_degree = std::min<size_t>(128, std::max<int64_t>(1, n_clusters - 1));

  index_params build_params;
  build_params.metric                   = params.metric;
  build_params.graph_degree             = graph_degree;
  build_params.intermediate_graph_degree = inter_degree;
  build_params.attach_dataset_on_build = true;

  // Build CAGRA index on centroids (centroids are [n_clusters, dim])
  auto centers_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    centroids.data_handle(), n_clusters, dim);
  auto cagra_index = cuvs::neighbors::cagra::build(handle, build_params, centers_view);

  // Queries: convert X to float if needed
  rmm::device_uvector<float> queries_buf(0, stream);
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries_view(nullptr, 0, 0);
  if constexpr (std::is_same_v<DataT, float>) {
    queries_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      reinterpret_cast<const float*>(X.data_handle()), n_rows, dim);
  } else {
    queries_buf.resize(static_cast<size_t>(n_rows) * static_cast<size_t>(dim), stream);
    auto queries_mat = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
      queries_buf.data(), n_rows, dim);
    raft::linalg::map(handle, raft::make_const_mdspan(X), queries_mat, mapping_op);
    queries_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      queries_buf.data(), n_rows, dim);
  }

  // Search k=1
  search_params search_params;
  search_params.max_queries = 0;  // auto

  auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(handle, n_rows, 1);
  auto distances = raft::make_device_matrix<float, int64_t>(handle, n_rows, 1);

  cuvs::neighbors::cagra::search(
    handle, search_params, cagra_index, queries_view, neighbors.view(), distances.view());

  // Copy neighbor indices (column 0) to labels with cast to LabelT
  auto neighbors_col = raft::make_device_vector_view<const uint32_t, int64_t>(
    neighbors.data_handle(), n_rows);
  auto labels_view = raft::make_device_vector_view<LabelT, IdxT>(labels.data_handle(), n_rows);
  raft::linalg::map(
    handle, raft::make_const_mdspan(neighbors_col), labels_view, raft::cast_op<LabelT>());
}

}  // namespace cuvs::cluster::kmeans::detail
