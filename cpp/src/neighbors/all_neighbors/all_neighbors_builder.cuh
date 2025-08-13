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
#include "raft/core/device_mdarray.hpp"
#include "raft/core/mdspan.hpp"
#include "raft/util/cudart_utils.hpp"
#include <cuvs/neighbors/all_neighbors.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <cuvs/neighbors/refine.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/managed_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>

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
    std::optional<raft::device_matrix_view<IdxT, IdxT, row_major>> indices = std::nullopt,
    std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances  = std::nullopt)
    : res{res},
      n_clusters{n_clusters},
      min_cluster_size{min_cluster_size},
      max_cluster_size{max_cluster_size},
      k{k},
      indices_{indices},
      distances_{distances}
  {
    RAFT_EXPECTS(this->n_clusters > 1 || indices.has_value(),
                 "indices should be preallocated to create knn builder for n_clusters == 1 (no "
                 "batching mode)");
    if (n_clusters > 1) {  // allocating additional space needed for batching
      inverted_indices_d.emplace(raft::make_device_vector<IdxT, IdxT>(res, max_cluster_size));
      batch_neighbors_h.emplace(raft::make_host_matrix<IdxT, IdxT>(max_cluster_size, k));
      batch_neighbors_d.emplace(raft::make_device_matrix<IdxT, IdxT>(res, max_cluster_size, k));
      batch_distances_d.emplace(raft::make_device_matrix<T, IdxT>(res, max_cluster_size, k));
    }
  }

  /**
   * Some memory-heavy allocations that can be used over multiple clusters should be allocated here
   * Arguments:
   * - [in] dataset: host_matrix_view or device_matrix_view of the the ENTIRE dataset
   */
  virtual void prepare_build(raft::host_matrix_view<const T, IdxT, row_major> dataset) {}
  virtual void prepare_build(raft::device_matrix_view<const T, IdxT, row_major> dataset) {}

  /**
   * Running the ann algorithm on the given cluster, and merging it into the global result
   * Arguments:
   * - [in] dataset: host_matrix_view or device_matrix_view of the cluster dataset
   * - [in] inverted_indices (optional): global data indices for the data points in the current
   * cluster of size [num_data_in_cluster]. Only needed when using the batching algorithm.
   * - [out] global_neighbors (optional): raft::managed_matrix_view type of [total_num_rows, k] for
   * final all-neighbors graph indices. Only needed when using the batching algorithm.
   * - [out] global_distances (optional): raft::managed_matrix_view type of [total_num_rows, k] for
   * final all-neighbors graph distances. Only needed when using the batching algorithm.
   */
  virtual void build_knn(
    raft::host_matrix_view<const T, IdxT, row_major> dataset,
    std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices    = std::nullopt,
    std::optional<raft::managed_matrix_view<IdxT, IdxT>> global_neighbors = std::nullopt,
    std::optional<raft::managed_matrix_view<T, IdxT>> global_distances    = std::nullopt)
  {
  }

  virtual void build_knn(raft::device_matrix_view<const T, IdxT, row_major> dataset) {}

  virtual ~all_neighbors_builder() = default;

  raft::resources const& res;
  size_t n_clusters, min_cluster_size, max_cluster_size, k;

  // these are optional types since we do not know the size at time of all_neighbors_builder
  // construction
  std::optional<raft::device_vector<IdxT, IdxT>> inverted_indices_d;
  std::optional<raft::host_matrix<IdxT, IdxT>> batch_neighbors_h;
  std::optional<raft::device_matrix<IdxT, IdxT>> batch_neighbors_d;
  std::optional<raft::device_matrix<T, IdxT>> batch_distances_d;

  // optional indices and distances, only used when n_clusters=1
  // when n_clusters > 1 (i.e. doing batching), we write to the global_neighbors and
  // global_distances managed memory arrays.
  std::optional<raft::device_matrix_view<IdxT, IdxT, row_major>> indices_;
  std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances_;
};

template <typename T, typename IdxT = int64_t>
struct all_neighbors_builder_ivfpq : public all_neighbors_builder<T, IdxT> {
  all_neighbors_builder_ivfpq(
    raft::resources const& res,
    size_t n_clusters,
    size_t min_cluster_size,
    size_t max_cluster_size,
    size_t k,
    graph_build_params::ivf_pq_params& params,
    std::optional<raft::device_matrix_view<IdxT, IdxT, row_major>> indices = std::nullopt,
    std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances  = std::nullopt)
    : all_neighbors_builder<T, IdxT>(
        res, n_clusters, min_cluster_size, max_cluster_size, k, indices, distances),
      all_ivf_pq_params{params}
  {
  }

  void prepare_build_common(size_t num_cols)
  {
    candidate_k = std::min<IdxT>(
      std::max(static_cast<size_t>(this->k * all_ivf_pq_params.refinement_rate), this->k),
      this->min_cluster_size);

    candidate_distances_d.emplace(
      raft::make_device_matrix<T, IdxT, row_major>(this->res, this->max_cluster_size, candidate_k));
    candidate_neighbors_d.emplace(raft::make_device_matrix<IdxT, IdxT, row_major>(
      this->res, this->max_cluster_size, candidate_k));
    candidate_neighbors_h.emplace(
      raft::make_host_matrix<IdxT, IdxT, row_major>(this->max_cluster_size, candidate_k));

    refined_neighbors_h.emplace(
      raft::make_host_matrix<IdxT, IdxT, row_major>(this->max_cluster_size, this->k));
    refined_distances_h.emplace(
      raft::make_host_matrix<T, IdxT, row_major>(this->max_cluster_size, this->k));
  }

  void prepare_build(raft::host_matrix_view<const T, IdxT, row_major> dataset) override
  {
    // if dataset is on host, then allocate space for device data because ivfpq requires data to be
    // on device
    size_t num_cols = dataset.extent(1);
    data_d.emplace(
      raft::make_device_matrix<T, IdxT, row_major>(this->res, this->max_cluster_size, num_cols));
    prepare_build_common(num_cols);
  }

  void prepare_build(raft::device_matrix_view<const T, IdxT, row_major> dataset) override
  {
    prepare_build_common(dataset.extent(1));
  }

  // Actual build logic using ivfpq.
  // need device and host views of the dataset because ivfpq build and search uses the device view,
  // and refine uses the host view
  void build_knn_common(
    raft::device_matrix_view<const T, IdxT, row_major> dataset_d,
    raft::host_matrix_view<const T, IdxT, row_major> dataset_h,
    std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices    = std::nullopt,
    std::optional<raft::managed_matrix_view<IdxT, IdxT>> global_neighbors = std::nullopt,
    std::optional<raft::managed_matrix_view<T, IdxT>> global_distances    = std::nullopt)
  {
    RAFT_EXPECTS(
      this->n_clusters <= 1 || (inverted_indices.has_value() && global_neighbors.has_value() &&
                                global_distances.has_value()),
      "need valid inverted_indices, global_neighbors, and global_distances for "
      "build_knn if doing batching.");

    size_t num_data_in_cluster = dataset_d.extent(0);
    size_t num_cols            = dataset_d.extent(1);

    auto index_ivfpq = ivf_pq::build(this->res, all_ivf_pq_params.build_params, dataset_d);

    auto candidate_distances_view = raft::make_device_matrix_view<T, IdxT>(
      candidate_distances_d.value().data_handle(), num_data_in_cluster, candidate_k);
    auto candidate_neighbors_view = raft::make_device_matrix_view<IdxT, IdxT>(
      candidate_neighbors_d.value().data_handle(), num_data_in_cluster, candidate_k);
    cuvs::neighbors::ivf_pq::search(this->res,
                                    all_ivf_pq_params.search_params,
                                    index_ivfpq,
                                    dataset_d,
                                    candidate_neighbors_view,
                                    candidate_distances_view);

    // copy candidate neighbors to host
    raft::copy(candidate_neighbors_h.value().data_handle(),
               candidate_neighbors_view.data_handle(),
               num_data_in_cluster * candidate_k,
               raft::resource::get_cuda_stream(this->res));
    auto candidate_neighbors_h_view = raft::make_host_matrix_view<IdxT, IdxT>(
      candidate_neighbors_h.value().data_handle(), num_data_in_cluster, candidate_k);
    auto refined_distances_h_view = raft::make_host_matrix_view<T, IdxT>(
      refined_distances_h.value().data_handle(), num_data_in_cluster, this->k);
    auto refined_neighbors_h_view = raft::make_host_matrix_view<IdxT, IdxT>(
      refined_neighbors_h.value().data_handle(), num_data_in_cluster, this->k);

    refine(this->res,
           dataset_h,
           dataset_h,
           raft::make_const_mdspan(candidate_neighbors_h_view),
           refined_neighbors_h_view,
           refined_distances_h_view,
           all_ivf_pq_params.build_params.metric);

    if (this->n_clusters > 1) {  // do batching
      raft::copy(this->batch_distances_d.value().data_handle(),
                 refined_distances_h_view.data_handle(),
                 num_data_in_cluster * this->k,
                 raft::resource::get_cuda_stream(this->res));

      remap_and_merge_subgraphs<T, IdxT, IdxT>(
        this->res,
        this->inverted_indices_d.value().view(),
        inverted_indices.value(),
        refined_neighbors_h.value().view(),
        this->batch_neighbors_h.value().view(),
        this->batch_neighbors_d.value().view(),
        this->batch_distances_d.value().view(),
        global_neighbors.value(),
        global_distances.value(),
        num_data_in_cluster,
        this->k,
        cuvs::distance::is_min_close(all_ivf_pq_params.build_params.metric));
    } else {
      size_t num_rows = num_data_in_cluster;
      // copy resulting indices and distances to device output
      raft::copy(this->indices_.value().data_handle(),
                 refined_neighbors_h_view.data_handle(),
                 num_rows * this->k,
                 raft::resource::get_cuda_stream(this->res));
      if (this->distances_.has_value()) {
        raft::copy(this->distances_.value().data_handle(),
                   refined_distances_h_view.data_handle(),
                   num_rows * this->k,
                   raft::resource::get_cuda_stream(this->res));
      }
    }
  }

  void build_knn(
    raft::host_matrix_view<const T, IdxT, row_major> dataset,
    std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices    = std::nullopt,
    std::optional<raft::managed_matrix_view<IdxT, IdxT>> global_neighbors = std::nullopt,
    std::optional<raft::managed_matrix_view<T, IdxT>> global_distances    = std::nullopt) override
  {
    // we need data on device for ivfpq build and search.
    raft::copy(data_d.value().data_handle(),
               dataset.data_handle(),
               dataset.size(),
               raft::resource::get_cuda_stream(this->res));

    build_knn_common(raft::make_device_matrix_view<const T, IdxT, row_major>(
                       data_d.value().data_handle(), dataset.extent(0), dataset.extent(1)),
                     dataset,
                     inverted_indices,
                     global_neighbors,
                     global_distances);
  }

  void build_knn(raft::device_matrix_view<const T, IdxT, row_major> dataset) override
  {
    RAFT_EXPECTS(this->n_clusters <= 1,
                 "building all-neighbors knn graph with dataset on device is not supported with "
                 "batching (n_clusters > 1)");

    // we allocate host memory here and not in the prepare_build function because this function is
    // not called for batching
    auto dataset_h = raft::make_host_matrix<T, IdxT>(dataset.extent(0), dataset.extent(1));

    // we need data on host for refining
    raft::copy(dataset_h.data_handle(),
               dataset.data_handle(),
               dataset.size(),
               raft::resource::get_cuda_stream(this->res));

    build_knn_common(dataset,
                     raft::make_host_matrix_view<const T, IdxT, row_major>(
                       dataset_h.data_handle(), dataset.extent(0), dataset.extent(1)));
  }

  graph_build_params::ivf_pq_params all_ivf_pq_params;
  size_t candidate_k;

  std::optional<raft::device_matrix<T, IdxT>> data_d;

  std::optional<raft::device_matrix<T, IdxT>> candidate_distances_d;
  std::optional<raft::device_matrix<IdxT, IdxT>> candidate_neighbors_d;
  std::optional<raft::host_matrix<IdxT, IdxT>> candidate_neighbors_h;

  std::optional<raft::host_matrix<IdxT, IdxT>> refined_neighbors_h;
  std::optional<raft::host_matrix<T, IdxT>> refined_distances_h;
};

template <typename T, typename IdxT = int64_t>
struct all_neighbors_builder_nn_descent : public all_neighbors_builder<T, IdxT> {
  all_neighbors_builder_nn_descent(
    raft::resources const& res,
    size_t n_clusters,
    size_t min_cluster_size,
    size_t max_cluster_size,
    size_t k,
    graph_build_params::nn_descent_params& params,
    std::optional<raft::device_matrix_view<IdxT, IdxT, row_major>> indices = std::nullopt,
    std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances  = std::nullopt)
    : all_neighbors_builder<T, IdxT>(
        res, n_clusters, min_cluster_size, max_cluster_size, k, indices, distances),
      nnd_params{params}
  {
  }

  template <typename Accessor>
  void prepare_build_common(mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> dataset)
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

    auto build_config                = nn_descent::detail::get_build_config(this->res,
                                                             nnd_params,
                                                             this->max_cluster_size,
                                                             static_cast<size_t>(dataset.extent(1)),
                                                             nnd_params.metric,
                                                             extended_graph_degree,
                                                             graph_degree);
    build_config.output_graph_degree = this->k;
    nnd_builder.emplace(this->res, build_config);
    int_graph.emplace(raft::make_host_matrix<int, IdxT, row_major>(
      this->max_cluster_size, static_cast<IdxT>(extended_graph_degree)));
  }

  void prepare_build(raft::device_matrix_view<const T, IdxT, row_major> dataset) override
  {
    prepare_build_common(dataset);
  }

  void prepare_build(raft::host_matrix_view<const T, IdxT, row_major> dataset) override
  {
    prepare_build_common(dataset);
  }

  template <typename Accessor>
  void build_knn_common(
    mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> dataset,
    std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices    = std::nullopt,
    std::optional<raft::managed_matrix_view<IdxT, IdxT>> global_neighbors = std::nullopt,
    std::optional<raft::managed_matrix_view<T, IdxT>> global_distances    = std::nullopt)
  {
    RAFT_EXPECTS(
      this->n_clusters <= 1 || (inverted_indices.has_value() && global_neighbors.has_value() &&
                                global_distances.has_value()),
      "need valid inverted_indices, global_neighbors, and global_distances for "
      "build_knn if doing batching.");

    if (this->n_clusters > 1) {
      bool return_distances      = true;
      size_t num_data_in_cluster = dataset.extent(0);
      nnd_builder.value().build(dataset.data_handle(),
                                static_cast<int>(num_data_in_cluster),
                                int_graph.value().data_handle(),
                                return_distances,
                                this->batch_distances_d.value().data_handle());

      remap_and_merge_subgraphs<T, IdxT, int>(this->res,
                                              this->inverted_indices_d.value().view(),
                                              inverted_indices.value(),
                                              int_graph.value().view(),
                                              this->batch_neighbors_h.value().view(),
                                              this->batch_neighbors_d.value().view(),
                                              this->batch_distances_d.value().view(),
                                              global_neighbors.value(),
                                              global_distances.value(),
                                              num_data_in_cluster,
                                              this->k,
                                              cuvs::distance::is_min_close(nnd_params.metric));
    } else {
      size_t num_rows = dataset.extent(0);

      nnd_builder.value().build(
        dataset.data_handle(),
        static_cast<int>(num_rows),
        int_graph.value().data_handle(),
        this->distances_.has_value(),
        this->distances_.value_or(raft::make_device_matrix<T, IdxT>(this->res, 0, 0).view())
          .data_handle());

      auto tmp_indices = raft::make_host_matrix<IdxT, IdxT>(int_graph.value().extent(0), this->k);

      // host slice
#pragma omp parallel for
      for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < this->k; j++) {
          tmp_indices(i, j) = static_cast<IdxT>(int_graph.value()(i, j));
        }
      }

      // copy to final device output
      raft::copy(this->indices_.value().data_handle(),
                 tmp_indices.data_handle(),
                 tmp_indices.extent(0) * this->k,
                 raft::resource::get_cuda_stream(this->res));
    }
  }

  void build_knn(
    raft::host_matrix_view<const T, IdxT> dataset,
    std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices    = std::nullopt,
    std::optional<raft::managed_matrix_view<IdxT, IdxT>> global_neighbors = std::nullopt,
    std::optional<raft::managed_matrix_view<T, IdxT>> global_distances    = std::nullopt) override
  {
    build_knn_common(dataset, inverted_indices, global_neighbors, global_distances);
  }

  void build_knn(raft::device_matrix_view<const T, IdxT> dataset) override
  {
    RAFT_EXPECTS(this->n_clusters <= 1,
                 "building all-neighbors knn graph with dataset on device is not supported with "
                 "batching (n_clusters > 1)");
    build_knn_common(dataset);
  }

  nn_descent::index_params nnd_params;
  nn_descent::detail::BuildConfig build_config;

  std::optional<nn_descent::detail::GNND<const T, int>> nnd_builder;
  std::optional<raft::host_matrix<int, IdxT>> int_graph;
};

template <typename T, typename IdxT = int64_t>
struct all_neighbors_builder_brute_force : public all_neighbors_builder<T, IdxT> {
  all_neighbors_builder_brute_force(
    raft::resources const& res,
    size_t n_clusters,
    size_t min_cluster_size,
    size_t max_cluster_size,
    size_t k,
    graph_build_params::brute_force_params& params,
    std::optional<raft::device_matrix_view<IdxT, IdxT, row_major>> indices = std::nullopt,
    std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances  = std::nullopt)
    : all_neighbors_builder<T, IdxT>(
        res, n_clusters, min_cluster_size, max_cluster_size, k, indices, distances),
      bf_params{params}
  {
  }

  void prepare_build(raft::device_matrix_view<const T, IdxT, row_major> dataset) override {}

  void prepare_build(raft::host_matrix_view<const T, IdxT, row_major> dataset) override
  {
    // data needs to be on device for build - search
    size_t num_cols = dataset.extent(1);
    data_d.emplace(
      raft::make_device_matrix<T, IdxT, row_major>(this->res, this->max_cluster_size, num_cols));
  }

  void build_knn_common(
    raft::device_matrix_view<const T, IdxT> dataset,
    std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices    = std::nullopt,
    std::optional<raft::managed_matrix_view<IdxT, IdxT>> global_neighbors = std::nullopt,
    std::optional<raft::managed_matrix_view<T, IdxT>> global_distances    = std::nullopt)
  {
    RAFT_EXPECTS(
      this->n_clusters <= 1 || (inverted_indices.has_value() && global_neighbors.has_value() &&
                                global_distances.has_value()),
      "need valid inverted_indices, global_neighbors, and global_distances for "
      "build_knn if doing batching.");

    if (this->n_clusters > 1) {
      auto idx = cuvs::neighbors::brute_force::build(this->res, bf_params.build_params, dataset);

      size_t num_data_in_cluster = dataset.extent(0);

      cuvs::neighbors::brute_force::search(
        this->res,
        bf_params.search_params,
        idx,
        dataset,
        raft::make_device_matrix_view<IdxT, IdxT>(
          this->batch_neighbors_d.value().data_handle(), num_data_in_cluster, this->k),
        raft::make_device_matrix_view<T, IdxT>(
          this->batch_distances_d.value().data_handle(), num_data_in_cluster, this->k));

      raft::copy(this->batch_neighbors_h.value().data_handle(),
                 this->batch_neighbors_d.value().data_handle(),
                 num_data_in_cluster * this->k,
                 raft::resource::get_cuda_stream(this->res));

      remap_and_merge_subgraphs<T, IdxT, IdxT>(
        this->res,
        this->inverted_indices_d.value().view(),
        inverted_indices.value(),
        this->batch_neighbors_h.value().view(),
        this->batch_neighbors_h.value().view(),
        this->batch_neighbors_d.value().view(),
        this->batch_distances_d.value().view(),
        global_neighbors.value(),
        global_distances.value(),
        num_data_in_cluster,
        this->k,
        cuvs::distance::is_min_close(bf_params.build_params.metric));
    } else {
      auto idx = cuvs::neighbors::brute_force::build(this->res, bf_params.build_params, dataset);

      cuvs::neighbors::brute_force::search(
        this->res,
        bf_params.search_params,
        idx,
        dataset,
        this->indices_.value(),
        this->distances_.has_value()
          ? this->distances_.value()
          : raft::make_device_matrix<T, IdxT>(this->res, dataset.extent(0), this->k).view());
    }
  }

  void build_knn(
    raft::host_matrix_view<const T, IdxT> dataset,
    std::optional<raft::host_vector_view<IdxT, IdxT>> inverted_indices    = std::nullopt,
    std::optional<raft::managed_matrix_view<IdxT, IdxT>> global_neighbors = std::nullopt,
    std::optional<raft::managed_matrix_view<T, IdxT>> global_distances    = std::nullopt) override
  {
    raft::copy(data_d.value().data_handle(),
               dataset.data_handle(),
               dataset.size(),
               raft::resource::get_cuda_stream(this->res));

    build_knn_common(raft::make_device_matrix_view<const T, IdxT, row_major>(
                       data_d.value().data_handle(), dataset.extent(0), dataset.extent(1)),
                     inverted_indices,
                     global_neighbors,
                     global_distances);
  }

  void build_knn(raft::device_matrix_view<const T, IdxT> dataset) override
  {
    RAFT_EXPECTS(this->n_clusters <= 1,
                 "building all-neighbors knn graph with dataset on device is not supported with "
                 "batching (n_clusters > 1)");
    build_knn_common(dataset);
  }

  graph_build_params::brute_force_params bf_params;

  std::optional<raft::device_matrix<T, IdxT, row_major>> data_d;
};

template <typename T, typename IdxT>
std::unique_ptr<all_neighbors_builder<T, IdxT>> get_knn_builder(
  const raft::resources& handle,
  const all_neighbors_params& params,
  size_t min_cluster_size,
  size_t max_cluster_size,
  size_t k,
  std::optional<raft::device_matrix_view<IdxT, IdxT, row_major>> indices = std::nullopt,
  std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances  = std::nullopt)
{
  if (std::holds_alternative<graph_build_params::brute_force_params>(params.graph_build_params)) {
    auto brute_force_params =
      std::get<graph_build_params::brute_force_params>(params.graph_build_params);
    if (brute_force_params.build_params.metric != params.metric) {
      RAFT_LOG_WARN("Setting brute_force_params metric to metric given for batching algorithm");
      brute_force_params.build_params.metric = params.metric;
    }
    return std::make_unique<all_neighbors_builder_brute_force<T, IdxT>>(handle,
                                                                        params.n_clusters,
                                                                        min_cluster_size,
                                                                        max_cluster_size,
                                                                        k,
                                                                        brute_force_params,
                                                                        indices,
                                                                        distances);
  } else if (std::holds_alternative<graph_build_params::nn_descent_params>(
               params.graph_build_params)) {
    auto nn_descent_params =
      std::get<graph_build_params::nn_descent_params>(params.graph_build_params);
    if (nn_descent_params.metric != params.metric) {
      RAFT_LOG_WARN("Setting nnd_params metric to metric given for batching algorithm");
      nn_descent_params.metric = params.metric;
    }
    return std::make_unique<all_neighbors_builder_nn_descent<T, IdxT>>(handle,
                                                                       params.n_clusters,
                                                                       min_cluster_size,
                                                                       max_cluster_size,
                                                                       k,
                                                                       nn_descent_params,
                                                                       indices,
                                                                       distances);
  } else if (std::holds_alternative<graph_build_params::ivf_pq_params>(params.graph_build_params)) {
    auto ivf_pq_params = std::get<graph_build_params::ivf_pq_params>(params.graph_build_params);
    if (ivf_pq_params.build_params.metric != params.metric) {
      RAFT_LOG_WARN("Setting ivfpq_params metric to metric given for batching algorithm");
      ivf_pq_params.build_params.metric = params.metric;
    }
    return std::make_unique<all_neighbors_builder_ivfpq<T, IdxT>>(handle,
                                                                  params.n_clusters,
                                                                  min_cluster_size,
                                                                  max_cluster_size,
                                                                  k,
                                                                  ivf_pq_params,
                                                                  indices,
                                                                  distances);
  } else {
    RAFT_FAIL("Batch KNN build algos only supporting Brute Force, NN Descent, and IVFPQ");
  }
}

}  // namespace cuvs::neighbors::all_neighbors::detail
