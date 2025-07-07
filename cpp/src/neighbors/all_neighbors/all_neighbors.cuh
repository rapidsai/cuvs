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
#include "../detail/reachability.cuh"
#include "all_neighbors_batched.cuh"
#include <cuvs/neighbors/all_neighbors.hpp>
#include <raft/matrix/shift.cuh>
#include <raft/util/cudart_utils.hpp>

namespace cuvs::neighbors::all_neighbors::detail {
using namespace cuvs::neighbors;

void check_metric(const all_neighbors_params& params, bool do_mutual_reachability_dist)
{
  if (std::holds_alternative<graph_build_params::brute_force_params>(params.graph_build_params)) {
    if (do_mutual_reachability_dist) {
      // TODO raft expects for this!
      auto allowed_metrics = params.metric == cuvs::distance::DistanceType::L2Expanded ||
                             params.metric == cuvs::distance::DistanceType::L2SqrtExpanded;
      RAFT_EXPECTS(allowed_metrics, "error message comes here");
    } else {
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
    }
  } else if (std::holds_alternative<graph_build_params::nn_descent_params>(
               params.graph_build_params)) {
    bool allowed_metrics = false;
    if (do_mutual_reachability_dist) {
      // InnerProduct is not supported for mutual reachability distance, because mutual reachability
      // distance takes "max" of core distances and pairwise distance.
      allowed_metrics = params.metric == cuvs::distance::DistanceType::L2Expanded ||
                        params.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                        params.metric == cuvs::distance::DistanceType::CosineExpanded;
    } else {
      allowed_metrics = params.metric == cuvs::distance::DistanceType::L2Expanded ||
                        params.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                        params.metric == cuvs::distance::DistanceType::CosineExpanded ||
                        params.metric == cuvs::distance::DistanceType::InnerProduct;
    }

    RAFT_EXPECTS(allowed_metrics,
                 "Distance metric for all-neighbors build with NN Descent should be L2Expanded, "
                 "L2SqrtExpanded, CosineExpanded, or InnerProduct. For mutual reachability "
                 "distance calculation, InnerProduct metric is not supported.");
  } else if (std::holds_alternative<graph_build_params::ivf_pq_params>(params.graph_build_params)) {
    RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded,
                 "Distance metric for all-neighbors build with IVFPQ should be L2Expanded");
    RAFT_EXPECTS(!do_mutual_reachability_dist,
                 "mutual reachability distance cannot be calculated using IVFPQ");
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
  std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances      = std::nullopt,
  std::optional<raft::device_vector_view<T, IdxT, row_major>> core_distances = std::nullopt,
  T alpha                                                                    = 1.0,
  bool include_self                                                          = false)
{
  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t num_cols = static_cast<size_t>(dataset.extent(1));

  auto knn_builder = get_knn_builder<T, IdxT>(
    handle, params, num_rows, num_rows, indices.extent(1), indices, distances);

  knn_builder->prepare_build(dataset);
  knn_builder->build_knn(dataset);
  raft::print_device_vector("indices", indices.data_handle(), indices.extent(1), std::cout);

  if (include_self) {
    raft::matrix::shift(handle, indices, 1);
    ;
    raft::matrix::shift(handle, distances.value(), 1, std::make_optional(static_cast<T>(0.0)));
  }

  if (core_distances.has_value()) {
    size_t k = indices.extent(1);
    cuvs::neighbors::detail::reachability::core_distances<IdxT, T>(
      distances.value().data_handle(),
      k,
      k,
      num_rows,
      core_distances.value().data_handle(),
      raft::resource::get_cuda_stream(handle));

    if (params.metric == cuvs::distance::DistanceType::L2SqrtExpanded &&
        std::holds_alternative<graph_build_params::nn_descent_params>(params.graph_build_params)) {
      // comparison within nn descent for L2SqrtExpanded is done without applying sqrt.
      raft::linalg::map(handle,
                        core_distances.value(),
                        raft::sq_op{},
                        raft::make_const_mdspan(core_distances.value()));
    }

    using ReachabilityPP = cuvs::neighbors::detail::reachability::ReachabilityPostProcess<int, T>;
    auto dist_epilogue   = ReachabilityPP{core_distances.value().data_handle(), alpha, num_rows};
    auto knn_builder     = get_knn_builder<T, IdxT, ReachabilityPP>(
      handle, params, num_rows, num_rows, indices.extent(1), indices, distances, dist_epilogue);
    knn_builder->prepare_build(dataset);
    knn_builder->build_knn(dataset);

    if (include_self) {
      raft::matrix::shift(handle, indices, 1);
      raft::matrix::shift(handle,
                          distances.value(),
                          raft::make_device_matrix_view<const T, IdxT>(
                            core_distances.value().data_handle(), num_rows, 1));
    }
  }
}

template <typename T, typename IdxT>
void build(
  const raft::resources& handle,
  const all_neighbors_params& params,
  raft::host_matrix_view<const T, IdxT, row_major> dataset,
  raft::device_matrix_view<IdxT, IdxT, row_major> indices,
  std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances      = std::nullopt,
  std::optional<raft::device_vector_view<T, IdxT, row_major>> core_distances = std::nullopt,
  T alpha                                                                    = 1.0)
{
  check_metric(params, core_distances.has_value());

  RAFT_EXPECTS(dataset.extent(0) == indices.extent(0),
               "number of rows in dataset should be the same as number of rows in indices matrix");

  if (distances.has_value()) {
    RAFT_EXPECTS(indices.extent(0) == distances.value().extent(0) &&
                   indices.extent(1) == distances.value().extent(1),
                 "indices matrix and distances matrix has to be the same shape.");
  }

  if (core_distances.has_value()) {
    RAFT_EXPECTS(distances.has_value(),
                 "distances matrix should be allocated to get mutual reachability distance.");
  }

  if (params.n_clusters == 1) {
    single_build(handle, params, dataset, indices, distances, core_distances, alpha);
  } else {
    batch_build(handle, params, dataset, indices, distances, core_distances, alpha);
  }
}

template <typename T, typename IdxT>
void build(
  const raft::resources& handle,
  const all_neighbors_params& params,
  raft::device_matrix_view<const T, IdxT, row_major> dataset,
  raft::device_matrix_view<IdxT, IdxT, row_major> indices,
  std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances      = std::nullopt,
  std::optional<raft::device_vector_view<T, IdxT, row_major>> core_distances = std::nullopt,
  T alpha                                                                    = 1.0)
{
  check_metric(params, core_distances.has_value());

  RAFT_EXPECTS(dataset.extent(0) == indices.extent(0),
               "number of rows in dataset should be the same as number of rows in indices matrix");

  if (distances.has_value()) {
    RAFT_EXPECTS(indices.extent(0) == distances.value().extent(0) &&
                   indices.extent(1) == distances.value().extent(1),
                 "indices matrix and distances matrix has to be the same shape.");
  }

  if (core_distances.has_value()) {
    RAFT_EXPECTS(distances.has_value(),
                 "distances matrix should be allocated to get mutual reachability distance.");
  }

  if (params.n_clusters > 1) {
    RAFT_FAIL(
      "Batched all-neighbors build is not supported with data on device. Put data on host for "
      "batch build.");
  } else {
    single_build(handle, params, dataset, indices, distances, core_distances, alpha);
  }
}
}  // namespace cuvs::neighbors::all_neighbors::detail
