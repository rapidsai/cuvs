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
#include <cuvs/neighbors/graph_build_types.hpp>
#include <raft/matrix/shift.cuh>
#include <raft/util/cudart_utils.hpp>

namespace cuvs::neighbors::all_neighbors::detail {
using namespace cuvs::neighbors;

GRAPH_BUILD_ALGO check_params_validity(const all_neighbors_params& params,
                                       bool do_mutual_reachability_dist)
{
  if (std::holds_alternative<graph_build_params::brute_force_params>(params.graph_build_params)) {
    /* There are issues with tiled_brute_force_knn (mainly the pairwise_distances functions being
     * used). pairwise_distances returns different distance for different input shapes, making it
     * difficult to use for batched all-neighbors which depend on distance comparison to rule out
     * duplicate indices when merging. among configurations [single, batched] x [normal
     * distance, mutual reach distance] batched, mutual reach distance uses tiled_brute_force_knn,
     * and therefore is not supported. Note that when k > 64, the brute_force API also ends up using
     * tiled_brute_force_knn, which may result in issues. related issue:
     * https://github.com/rapidsai/cuvs/issues/1056
     */
    if (do_mutual_reachability_dist) {
      RAFT_EXPECTS(params.n_clusters <= 1,
                   "Batched all-neighbors build with brute force for getting mutual reachability "
                   "distances is not supported.");
      // InnerProduct is not supported for mutual reachability distance, because mutual reachability
      // distance takes "max" of core distances and pairwise distance.
      auto allowed_metrics = params.metric == cuvs::distance::DistanceType::L2Expanded ||
                             params.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                             params.metric == cuvs::distance::DistanceType::CosineExpanded;
      RAFT_EXPECTS(
        allowed_metrics,
        "Distance metric for all-neighbors build with brute force for computing mutual "
        "reachability distance should be L2Expanded, L2SqrtExpanded, or CosineExpanded.");
    } else {
      auto allowed_metrics_batch = params.metric == cuvs::distance::DistanceType::L2Expanded ||
                                   params.metric == cuvs::distance::DistanceType::L2SqrtExpanded;
      auto allowed_metrics_single = allowed_metrics_batch ||
                                    params.metric == cuvs::distance::DistanceType::CosineExpanded ||
                                    params.metric == cuvs::distance::DistanceType::InnerProduct;
      RAFT_EXPECTS((params.n_clusters <= 1 && allowed_metrics_single) || allowed_metrics_batch,
                   "Distance metric supported for for all-neighbors build with brute force depends "
                   "on params.n_clusters. When params.n_clusters <= 1, supported metrics are "
                   "L2Expanded, L2SqrtExpanded, CosineExpanded, or InnerProduct. When "
                   "params.n_clusters > 1, supported metrics are L2Expanded, L2SqrtExpanded.");
    }
    return GRAPH_BUILD_ALGO::BRUTE_FORCE;
  } else if (std::holds_alternative<graph_build_params::nn_descent_params>(
               params.graph_build_params)) {
    if (do_mutual_reachability_dist) {
      // InnerProduct is not supported for mutual reachability distance, because mutual reachability
      // distance takes "max" of core distances and pairwise distance.
      auto allowed_metrics = params.metric == cuvs::distance::DistanceType::L2Expanded ||
                             params.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                             params.metric == cuvs::distance::DistanceType::CosineExpanded;
      RAFT_EXPECTS(
        allowed_metrics,
        "Distance metric for all-neighbors build with NN Descent for computing mutual reachability "
        "distance should be L2Expanded, L2SqrtExpanded, or CosineExpanded.");
    } else {
      auto allowed_metrics = params.metric == cuvs::distance::DistanceType::L2Expanded ||
                             params.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                             params.metric == cuvs::distance::DistanceType::CosineExpanded ||
                             params.metric == cuvs::distance::DistanceType::InnerProduct;
      RAFT_EXPECTS(allowed_metrics,
                   "Distance metric for all-neighbors build with NN Descent should be L2Expanded, "
                   "L2SqrtExpanded, CosineExpanded, or InnerProduct.");
    }
    return GRAPH_BUILD_ALGO::NN_DESCENT;
  } else if (std::holds_alternative<graph_build_params::ivf_pq_params>(params.graph_build_params)) {
    RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded,
                 "Distance metric for all-neighbors build with IVFPQ should be L2Expanded");
    RAFT_EXPECTS(!do_mutual_reachability_dist,
                 "mutual reachability distance cannot be calculated using IVFPQ");
    return GRAPH_BUILD_ALGO::IVF_PQ;
  } else {
    RAFT_FAIL("Invalid all-neighbors build algo");
  }
}

// Single build (i.e. no batching) supports both host and device datasets
template <typename T, typename IdxT, typename Accessor, typename DistEpilogueT = raft::identity_op>
void single_build(
  const raft::resources& handle,
  const all_neighbors_params& params,
  mdspan<const T, matrix_extent<IdxT>, row_major, Accessor> dataset,
  raft::device_matrix_view<IdxT, IdxT, row_major> indices,
  std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances = std::nullopt,
  DistEpilogueT dist_epilogue                                           = DistEpilogueT{})
{
  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t num_cols = static_cast<size_t>(dataset.extent(1));

  auto knn_builder = get_knn_builder<T, IdxT>(
    handle, params, num_rows, num_rows, indices.extent(1), indices, distances, dist_epilogue);

  knn_builder->prepare_build(dataset);
  knn_builder->build_knn(dataset);
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
  auto build_algo = check_params_validity(params, core_distances.has_value());

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
    single_build(handle, params, dataset, indices, distances);
  } else {
    batch_build(handle, params, dataset, indices, distances);
  }

  if (core_distances.has_value()) {
    size_t k        = indices.extent(1);
    size_t num_rows = core_distances.value().size();
    cuvs::neighbors::detail::reachability::core_distances<IdxT, T>(
      distances.value().data_handle(),
      k,
      k,
      num_rows,
      core_distances.value().data_handle(),
      raft::resource::get_cuda_stream(handle));

    using ReachabilityPP = cuvs::neighbors::detail::reachability::ReachabilityPostProcess<IdxT, T>;
    auto dist_epilogue   = ReachabilityPP{core_distances.value().data_handle(), alpha, num_rows};
    if (params.n_clusters == 1) {
      single_build(handle, params, dataset, indices, distances, dist_epilogue);
    } else {
      batch_build(handle, params, dataset, indices, distances, dist_epilogue);
    }
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
  auto build_algo = check_params_validity(params, core_distances.has_value());

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
    single_build(handle, params, dataset, indices, distances);
  }

  if (core_distances.has_value()) {
    size_t k        = indices.extent(1);
    size_t num_rows = core_distances.value().size();
    cuvs::neighbors::detail::reachability::core_distances<IdxT, T>(
      distances.value().data_handle(),
      k,
      k,
      num_rows,
      core_distances.value().data_handle(),
      raft::resource::get_cuda_stream(handle));

    using ReachabilityPP = cuvs::neighbors::detail::reachability::ReachabilityPostProcess<IdxT, T>;
    auto dist_epilogue   = ReachabilityPP{core_distances.value().data_handle(), alpha, num_rows};
    single_build(handle, params, dataset, indices, distances, dist_epilogue);
  }
}
}  // namespace cuvs::neighbors::all_neighbors::detail
