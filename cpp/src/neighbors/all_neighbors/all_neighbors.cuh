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

namespace cuvs::neighbors::all_neighbors::detail {
using namespace cuvs::neighbors;

void check_metric(const index_params& params)
{
  if (std::holds_alternative<graph_build_params::nn_descent_params>(params.graph_build_params)) {
    auto allowed_metrics = params.metric == cuvs::distance::DistanceType::L2Expanded ||
                           params.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                           params.metric == cuvs::distance::DistanceType::CosineExpanded ||
                           params.metric == cuvs::distance::DistanceType::InnerProduct;
    RAFT_EXPECTS(allowed_metrics,
                 "Distance metric for all-neighbors build with NN Descent should be L2Expanded, "
                 "L2SqrtExpanded, CosineExpanded or "
                 "InnerProduct");
  } else if (std::holds_alternative<graph_build_params::ivf_pq_params>(params.graph_build_params)) {
    auto allowed_metrics = params.metric == cuvs::distance::DistanceType::L2Expanded ||
                           params.metric == cuvs::distance::DistanceType::InnerProduct;
    RAFT_EXPECTS(allowed_metrics,
                 "Distance metric for all-neighbors build with IVFPQ should be L2Expanded, or "
                 "InnerProduct");
  } else {
    RAFT_FAIL("Invalid all-neighbors build algo");
  }
}

// Supports both host and device datasets
template <typename T, typename IdxT, typename Accessor>
void single_build(const raft::resources& handle,
                  mdspan<const T, matrix_extent<IdxT>, row_major, Accessor> dataset,
                  const index_params& params,
                  all_neighbors::index<IdxT, T>& index)
{
  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t num_cols = static_cast<size_t>(dataset.extent(1));

  auto knn_builder = get_knn_builder<T, IdxT>(
    handle, params, static_cast<size_t>(index.k()), num_rows, num_rows, false);

  knn_builder->prepare_build(dataset);
  knn_builder->build_knn(params, dataset, index);
}

template <typename T, typename IdxT>
void build(const raft::resources& handle,
           raft::host_matrix_view<const T, IdxT, row_major> dataset,
           const index_params& params,
           all_neighbors::index<IdxT, T>& index)
{
  check_metric(params);

  if (params.n_clusters == 1) {
    single_build(handle, dataset, params, index);
  } else {
    batch_build(handle, dataset, params, index);
  }
}

template <typename T, typename IdxT = int64_t>
all_neighbors::index<IdxT, T> build(
  const raft::resources& handle,
  raft::host_matrix_view<const T, IdxT, row_major> dataset,
  int64_t k,
  const index_params& params,
  bool return_distances = false)  // distance type same as data type
{
  all_neighbors::index<IdxT, T> index{
    handle, static_cast<int64_t>(dataset.extent(0)), k, return_distances};
  build(handle, dataset, params, index);
  return index;
}

template <typename T, typename IdxT>
void build(const raft::resources& handle,
           raft::device_matrix_view<const T, IdxT, row_major> dataset,
           const index_params& params,
           all_neighbors::index<IdxT, T>& index)
{
  check_metric(params);

  if (params.n_clusters > 1) {
    RAFT_FAIL(
      "Batched all-neighbors build is not supported with data on device. Put data on host for "
      "batch build.");
  } else {
    single_build(handle, dataset, params, index);
  }
}

template <typename T, typename IdxT = int64_t>
all_neighbors::index<IdxT, T> build(
  const raft::resources& handle,
  raft::device_matrix_view<const T, IdxT, row_major> dataset,
  int64_t k,
  const index_params& params,
  bool return_distances = false)  // distance type same as data type
{
  check_metric(params);

  if (params.n_clusters > 1) {
    RAFT_FAIL(
      "Batched all-neighbors build is not supported with data on device. Put data on host for "
      "batch build.");
  } else {
    all_neighbors::index<IdxT, T> index{
      handle, static_cast<int64_t>(dataset.extent(0)), k, return_distances};
    single_build(handle, dataset, params, index);
    return index;
  }
}

}  // namespace cuvs::neighbors::all_neighbors::detail
