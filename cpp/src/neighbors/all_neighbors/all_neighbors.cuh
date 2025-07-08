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
#include "all_neighbors_batched.cuh"
#include <cuvs/neighbors/all_neighbors.hpp>
#include <raft/util/cudart_utils.hpp>

namespace cuvs::neighbors::all_neighbors::detail {
using namespace cuvs::neighbors;

void check_metric(const all_neighbors_params& params)
{
  if (std::holds_alternative<graph_build_params::brute_force_params>(params.graph_build_params)) {
    auto allowed_metrics_batch = params.metric == cuvs::distance::DistanceType::L2Expanded ||
                                 params.metric == cuvs::distance::DistanceType::L2SqrtExpanded;
    auto allowed_metrics_single = allowed_metrics_batch ||
                                  params.metric == cuvs::distance::DistanceType::CosineExpanded ||
                                  params.metric == cuvs::distance::DistanceType::InnerProduct;
    // related issue: https://github.com/rapidsai/cuvs/issues/1056
    RAFT_EXPECTS((params.n_clusters <= 1 && allowed_metrics_single) || allowed_metrics_batch,
                 "Distance metric supported for for all-neighbors build with brute force depends "
                 "on params.n_clusters. When params.n_clusters <= 1, supported metrics are "
                 "L2Expanded, L2SqrtExpanded, CosineExpanded, or InnerProduct. When "
                 "params.n_clusters > 1, supported metrics are L2Expanded, L2SqrtExpanded.");
  } else if (std::holds_alternative<graph_build_params::nn_descent_params>(
               params.graph_build_params)) {
    auto allowed_metrics = params.metric == cuvs::distance::DistanceType::L2Expanded ||
                           params.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                           params.metric == cuvs::distance::DistanceType::CosineExpanded ||
                           params.metric == cuvs::distance::DistanceType::InnerProduct;
    RAFT_EXPECTS(allowed_metrics,
                 "Distance metric for all-neighbors build with NN Descent should be L2Expanded, "
                 "L2SqrtExpanded, CosineExpanded, or InnerProduct");
  } else if (std::holds_alternative<graph_build_params::ivf_pq_params>(params.graph_build_params)) {
    RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded,
                 "Distance metric for all-neighbors build with IVFPQ should be L2Expanded");
  } else {
    RAFT_FAIL("Invalid all-neighbors build algo");
  }
}

// Single build (i.e. no batching) supports both host and device datasets
template <typename T, typename IdxT, typename Accessor>
void single_build(
  const raft::resources& handle,
  const all_neighbors_params& params,
  mdspan<const T, matrix_extent<IdxT>, row_major, Accessor> dataset,
  raft::device_matrix_view<IdxT, IdxT, row_major> indices,
  std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances = std::nullopt)
{
  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t num_cols = static_cast<size_t>(dataset.extent(1));

  auto knn_builder = get_knn_builder<T, IdxT>(
    handle, params, num_rows, num_rows, indices.extent(1), indices, distances);

  knn_builder->prepare_build(dataset);
  knn_builder->build_knn(dataset);
}

template <typename T, typename IdxT>
void build(const raft::resources& handle,
           const all_neighbors_params& params,
           raft::host_matrix_view<const T, IdxT, row_major> dataset,
           raft::device_matrix_view<IdxT, IdxT, row_major> indices,
           std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances = std::nullopt)
{
  check_metric(params);

  RAFT_EXPECTS(dataset.extent(0) == indices.extent(0),
               "number of rows in dataset should be the same as number of rows in indices matrix");

  if (distances.has_value()) {
    RAFT_EXPECTS(indices.extent(0) == distances.value().extent(0) &&
                   indices.extent(1) == distances.value().extent(1),
                 "indices matrix and distances matrix has to be the same shape.");
  }

  if (params.n_clusters == 1) {
    single_build(handle, params, dataset, indices, distances);
  } else {
    batch_build(handle, params, dataset, indices, distances);
  }
}

template <typename T, typename IdxT>
void build(const raft::resources& handle,
           const all_neighbors_params& params,
           raft::device_matrix_view<const T, IdxT, row_major> dataset,
           raft::device_matrix_view<IdxT, IdxT, row_major> indices,
           std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances = std::nullopt)
{
  check_metric(params);

  RAFT_EXPECTS(dataset.extent(0) == indices.extent(0),
               "number of rows in dataset should be the same as number of rows in indices matrix");

  if (distances.has_value()) {
    RAFT_EXPECTS(indices.extent(0) == distances.value().extent(0) &&
                   indices.extent(1) == distances.value().extent(1),
                 "indices matrix and distances matrix has to be the same shape.");
  }

  if (params.n_clusters > 1) {
    RAFT_FAIL(
      "Batched all-neighbors build is not supported with data on device. Put data on host for "
      "batch build.");
  } else {
    single_build(handle, params, dataset, indices, distances);
  }
}
}  // namespace cuvs::neighbors::all_neighbors::detail
