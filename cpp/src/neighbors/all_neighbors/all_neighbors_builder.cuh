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

#include "../detail/nn_descent_gnnd.hpp"
#include "all_neighbors_merge.cuh"
#include <cuvs/neighbors/all_neighbors.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <cuvs/neighbors/refine.hpp>

#include <raft/core/managed_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>

namespace cuvs::neighbors::all_neighbors::detail {
using namespace cuvs::neighbors;

template <typename T, typename IdxT = int64_t>
struct all_neighbors_builder {
  all_neighbors_builder(
    raft::resources const& res,
    size_t n_clusters,
    size_t min_cluster_size,
    size_t max_cluster_size,
    size_t k,
    bool do_batch,
    cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
    : res{res},
      k{k},
      n_clusters{n_clusters},
      min_cluster_size{min_cluster_size},
      max_cluster_size{max_cluster_size},
      do_batch{do_batch},
      metric{metric}
  {
    if (do_batch) {
      inverted_indices_d.emplace(raft::make_device_vector<IdxT, IdxT>(res, max_cluster_size));
      batch_neighbors_h.emplace(raft::make_host_matrix<IdxT, IdxT>(max_cluster_size, k));
      batch_neighbors_d.emplace(raft::make_device_matrix<IdxT, IdxT>(res, max_cluster_size, k));
      batch_distances_d.emplace(raft::make_device_matrix<T, IdxT>(res, max_cluster_size, k));
    }
  }

  /**
   * Some memory-heavy allocations that can be used over multiple clusters should be allocated here
   * Arguments:
   * - [in] dataset: host_matrix_view of the the ENTIRE dataset
   */
  virtual void prepare_build(raft::host_matrix_view<const T, IdxT, raft::row_major> dataset) {}

  /**
   * Running the ann algorithm on the given cluster, and merging it into the global result
   * Arguments:
   * - [in] res: raft resource
   * - [in] params: all_neighbors::index_params
   * - [in] dataset: host_matrix_view of the cluster dataset
   * - [in] index
   * - [in] inverted_indices (optional): global data indices for the data points in the current
   * cluster of size (num_data_in_cluster). Only needed when calling with the batching algorithm
   * - [out] global_neighbors (optional): raft::managed_matrix_view type of (total_num_rows, k) for
   * final all-neighbors graph indices. Only needed when calling with the batching algorithm
   * - [out] global_distances (optional): raft::managed_matrix_view type of (total_num_rows, k) for
   * final all-neighbors graph distances. Only needed when calling with the batching algorithm
   */
  virtual void build_knn(
    raft::resources const& res,
    const index_params& params,
    raft::host_matrix_view<const T, IdxT, row_major> dataset,
    all_neighbors::index<IdxT, T>& index,
    std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices    = std::nullopt,
    std::optional<raft::managed_matrix_view<IdxT, IdxT>> global_neighbors = std::nullopt,
    std::optional<raft::managed_matrix_view<T, IdxT>> global_distances    = std::nullopt)
  {
  }

  raft::resources const& res;
  size_t n_clusters, min_cluster_size, max_cluster_size, k;
  cuvs::distance::DistanceType metric;
  bool do_batch;

  std::optional<raft::device_vector<IdxT, IdxT>> inverted_indices_d;
  std::optional<raft::host_matrix<IdxT, IdxT>> batch_neighbors_h;
  std::optional<raft::device_matrix<IdxT, IdxT>> batch_neighbors_d;
  std::optional<raft::device_matrix<T, IdxT>> batch_distances_d;
};

template <typename T, typename IdxT = int64_t>
struct all_neighbors_builder_ivfpq : public all_neighbors_builder<T, IdxT> {
  all_neighbors_builder_ivfpq(raft::resources const& res,
                              size_t n_clusters,
                              size_t min_cluster_size,
                              size_t max_cluster_size,
                              size_t k,
                              bool do_batch,
                              cuvs::distance::DistanceType metric,
                              all_neighbors::graph_build_params::ivf_pq_params& params)
    : all_neighbors_builder<T, IdxT>(
        res, n_clusters, min_cluster_size, max_cluster_size, k, do_batch, metric),
      all_ivf_pq_params{params}
  {
    if (all_ivf_pq_params.build_params.metric != metric) {
      RAFT_LOG_WARN("Setting ivfpq_params metric to metric given for batching algorithm");
      all_ivf_pq_params.build_params.metric = metric;
    }
  }

  void prepare_build(raft::host_matrix_view<const T, IdxT, row_major> dataset) override
  {
    size_t num_cols = static_cast<size_t>(dataset.extent(1));
    candidate_k     = std::min<IdxT>(
      std::max(static_cast<size_t>(this->k * all_ivf_pq_params.refinement_rate), this->k),
      this->min_cluster_size);
    data_d.emplace(
      raft::make_device_matrix<T, IdxT, row_major>(this->res, this->max_cluster_size, num_cols));

    distances_candidate_d.emplace(
      raft::make_device_matrix<T, IdxT, row_major>(this->res, this->max_cluster_size, candidate_k));
    neighbors_candidate_d.emplace(raft::make_device_matrix<IdxT, IdxT, row_major>(
      this->res, this->max_cluster_size, candidate_k));
    neighbors_candidate_h.emplace(
      raft::make_host_matrix<IdxT, IdxT, row_major>(this->max_cluster_size, candidate_k));

    // for host refining
    if (this->do_batch) {
      refined_neighbors_h.emplace(
        raft::make_host_matrix<IdxT, IdxT, row_major>(this->max_cluster_size, this->k));
    }
    refined_distances_h.emplace(
      raft::make_host_matrix<T, IdxT, row_major>(this->max_cluster_size, this->k));
  }

  void build_knn(
    raft::resources const& res,
    const index_params& params,
    raft::host_matrix_view<const T, IdxT, row_major> dataset,
    all_neighbors::index<IdxT, T>& index,
    std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices    = std::nullopt,
    std::optional<raft::managed_matrix_view<IdxT, IdxT>> global_neighbors = std::nullopt,
    std::optional<raft::managed_matrix_view<T, IdxT>> global_distances    = std::nullopt) override
  {
    RAFT_EXPECTS(!this->do_batch || (inverted_indices.has_value() && global_neighbors.has_value() &&
                                     global_distances.has_value()),
                 "inverted_indices, gloabl_neighbors, and global_distances should be passed for "
                 "build_knn if doing batching.");

    size_t num_data_in_cluster = dataset.extent(0);
    size_t num_cols            = dataset.extent(1);

    // we need data on device for ivfpq build and search.
    // num_data_in_cluster is always <= max_cluster_size
    raft::copy(data_d.value().data_handle(),
               dataset.data_handle(),
               num_data_in_cluster * num_cols,
               raft::resource::get_cuda_stream(this->res));

    auto data_view = raft::make_device_matrix_view<const T, IdxT>(
      data_d.value().data_handle(), num_data_in_cluster, num_cols);

    auto index_ivfpq = ivf_pq::build(this->res, all_ivf_pq_params.build_params, data_view);

    auto distances_candidate_view = raft::make_device_matrix_view<T, IdxT>(
      distances_candidate_d.value().data_handle(), num_data_in_cluster, candidate_k);
    auto neighbors_candidate_view = raft::make_device_matrix_view<IdxT, IdxT>(
      neighbors_candidate_d.value().data_handle(), num_data_in_cluster, candidate_k);
    cuvs::neighbors::ivf_pq::search(this->res,
                                    all_ivf_pq_params.search_params,
                                    index_ivfpq,
                                    data_view,
                                    neighbors_candidate_view,
                                    distances_candidate_view);
    raft::copy(neighbors_candidate_h.value().data_handle(),
               neighbors_candidate_view.data_handle(),
               num_data_in_cluster * candidate_k,
               raft::resource::get_cuda_stream(this->res));

    auto neighbors_candidate_h_view = raft::make_host_matrix_view<IdxT, IdxT>(
      neighbors_candidate_h.value().data_handle(), num_data_in_cluster, candidate_k);
    auto refined_distances_h_view = raft::make_host_matrix_view<T, IdxT>(
      refined_distances_h.value().data_handle(), num_data_in_cluster, this->k);

    raft::host_matrix_view<IdxT, IdxT> refined_neighbors_h_view;
    if (this->do_batch) {
      refined_neighbors_h_view = raft::make_host_matrix_view<IdxT, IdxT>(
        refined_neighbors_h.value().data_handle(), num_data_in_cluster, this->k);
    } else {
      refined_neighbors_h_view = index.graph();
    }

    refine(this->res,
           dataset,
           dataset,
           raft::make_const_mdspan(neighbors_candidate_h_view),
           refined_neighbors_h_view,
           refined_distances_h_view,
           params.metric);

    if (this->do_batch) {
      raft::copy(this->batch_distances_d.value().data_handle(),
                 refined_distances_h_view.data_handle(),
                 num_data_in_cluster * this->k,
                 raft::resource::get_cuda_stream(this->res));

      remap_and_merge_subgraphs<T, IdxT, IdxT>(this->res,
                                               this->inverted_indices_d.value().view(),
                                               inverted_indices.value(),
                                               refined_neighbors_h.value().view(),
                                               this->batch_neighbors_h.value().view(),
                                               this->batch_neighbors_d.value().view(),
                                               this->batch_distances_d.value().view(),
                                               global_neighbors.value(),
                                               global_distances.value(),
                                               num_data_in_cluster,
                                               this->k);
    } else {
      size_t num_rows = num_data_in_cluster;
      if (index.return_distances()) {
        raft::copy(index.distances().value().data_handle(),
                   refined_distances_h_view.data_handle(),
                   num_rows * this->k,
                   raft::resource::get_cuda_stream(this->res));
      }
    }
  }

  all_neighbors::graph_build_params::ivf_pq_params all_ivf_pq_params;
  size_t candidate_k;

  std::optional<raft::device_matrix<T, IdxT>> data_d;
  std::optional<raft::device_matrix<T, IdxT>> distances_candidate_d;
  std::optional<raft::device_matrix<IdxT, IdxT>> neighbors_candidate_d;
  std::optional<raft::host_matrix<IdxT, IdxT>> neighbors_candidate_h;
  std::optional<raft::host_matrix<IdxT, IdxT>> refined_neighbors_h;
  std::optional<raft::host_matrix<T, IdxT>> refined_distances_h;
};

template <typename T, typename IdxT = int64_t>
struct all_neighbors_builder_nn_descent : public all_neighbors_builder<T, IdxT> {
  all_neighbors_builder_nn_descent(raft::resources const& res,
                                   size_t n_clusters,
                                   size_t min_cluster_size,
                                   size_t max_cluster_size,
                                   size_t k,
                                   bool do_batch,
                                   cuvs::distance::DistanceType metric,
                                   all_neighbors::graph_build_params::nn_descent_params& params)
    : all_neighbors_builder<T, IdxT>(
        res, n_clusters, min_cluster_size, max_cluster_size, k, do_batch, metric),
      nnd_params{params}
  {
    if (nnd_params.metric != metric) {
      RAFT_LOG_WARN("Setting nnd_params metric to metric given for batching algorithm");
      nnd_params.metric = metric;
    }
  }

  void prepare_build(raft::host_matrix_view<const T, IdxT, row_major> dataset) override
  {
    if (nnd_params.graph_degree < this->k) {
      RAFT_LOG_WARN(
        "NN Descent's graph degree (%lu) has to be larger than or equal to k. Setting graph_degree "
        "to k (%lu).",
        nnd_params.graph_degree,
        this->k);
      nnd_params.graph_degree = this->k;
    }

    size_t extended_graph_degree, graph_degree;

    auto build_config = nn_descent::detail::get_build_config(
      this->res, nnd_params, dataset, nnd_params.metric, extended_graph_degree, graph_degree);
    build_config.output_graph_degree = this->k;
    nnd_builder.emplace(this->res, build_config);
    int_graph.emplace(raft::make_host_matrix<int, IdxT, row_major>(
      this->max_cluster_size, static_cast<IdxT>(extended_graph_degree)));
  }

  void build_knn(
    raft::resources const& res,
    const index_params& params,
    raft::host_matrix_view<const T, IdxT> dataset,
    all_neighbors::index<IdxT, T>& index,
    std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices    = std::nullopt,
    std::optional<raft::managed_matrix_view<IdxT, IdxT>> global_neighbors = std::nullopt,
    std::optional<raft::managed_matrix_view<T, IdxT>> global_distances    = std::nullopt) override
  {
    RAFT_EXPECTS(!this->do_batch || (inverted_indices.has_value() && global_neighbors.has_value() &&
                                     global_distances.has_value()),
                 "inverted_indices, gloabl_neighbors, and global_distances should be passed for "
                 "build_knn if doing batching.");

    if (this->do_batch) {
      // TODO add raft expect value for global and inverted
      bool return_distances      = true;
      size_t num_data_in_cluster = dataset.extent(0);
      nnd_builder.value().build(dataset.data_handle(),
                                static_cast<int>(num_data_in_cluster),
                                int_graph.value().data_handle(),
                                return_distances,
                                this->batch_distances_d.value().data_handle());

      remap_and_merge_subgraphs<T, IdxT, int>(res,
                                              this->inverted_indices_d.value().view(),
                                              inverted_indices.value(),
                                              int_graph.value().view(),
                                              this->batch_neighbors_h.value().view(),
                                              this->batch_neighbors_d.value().view(),
                                              this->batch_distances_d.value().view(),
                                              global_neighbors.value(),
                                              global_distances.value(),
                                              num_data_in_cluster,
                                              this->k);
    } else {
      size_t num_rows  = dataset.extent(0);
      T* distances_ptr = nullptr;
      if (index.return_distances()) { distances_ptr = index.distances().value().data_handle(); }
      nnd_builder.value().build(dataset.data_handle(),
                                static_cast<int>(num_rows),
                                int_graph.value().data_handle(),
                                index.return_distances(),
                                distances_ptr);

      // host slice
#pragma omp parallel for
      for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < this->k; j++) {
          index.graph()(i, j) = int_graph.value()(i, j);
        }
      }
    }
  }

  nn_descent::index_params nnd_params;
  nn_descent::detail::BuildConfig build_config;

  std::optional<nn_descent::detail::GNND<const T, int>> nnd_builder;
  std::optional<raft::host_matrix<int, IdxT>> int_graph;
};

template <typename T, typename IdxT>
std::unique_ptr<all_neighbors_builder<T, IdxT>> get_knn_builder(const raft::resources& handle,
                                                                const index_params& params,
                                                                size_t k,
                                                                size_t min_cluster_size,
                                                                size_t max_cluster_size,
                                                                bool do_batch)
{
  if (std::holds_alternative<graph_build_params::nn_descent_params>(params.graph_build_params)) {
    auto nn_descent_params =
      std::get<graph_build_params::nn_descent_params>(params.graph_build_params);

    return std::make_unique<all_neighbors_builder_nn_descent<T, IdxT>>(handle,
                                                                       params.n_clusters,
                                                                       min_cluster_size,
                                                                       max_cluster_size,
                                                                       k,
                                                                       do_batch,
                                                                       params.metric,
                                                                       nn_descent_params);
  } else if (std::holds_alternative<graph_build_params::ivf_pq_params>(params.graph_build_params)) {
    auto ivf_pq_params = std::get<graph_build_params::ivf_pq_params>(params.graph_build_params);
    return std::make_unique<all_neighbors_builder_ivfpq<T, IdxT>>(handle,
                                                                  params.n_clusters,
                                                                  min_cluster_size,
                                                                  max_cluster_size,
                                                                  k,
                                                                  do_batch,
                                                                  params.metric,
                                                                  ivf_pq_params);
  } else {
    RAFT_FAIL("Batch KNN build algos only supporting NN Descent and IVF PQ");
  }
}

}  // namespace cuvs::neighbors::all_neighbors::detail
