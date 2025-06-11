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
#include "../detail/reachability_types.cuh"
#include "all_neighbors_builder.cuh"
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/all_neighbors.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/refine.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/managed_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/matrix/sample_rows.cuh>
#include <raft/matrix/shift.cuh>
#include <raft/util/cudart_utils.hpp>
#include <variant>

namespace cuvs::neighbors::all_neighbors::detail {
using namespace cuvs::neighbors;

template <typename T, typename IdxT>
void reset_global_matrices(raft::resources const& res,
                           cuvs::distance::DistanceType metric,
                           raft::managed_matrix_view<IdxT, IdxT> global_neighbors,
                           raft::managed_matrix_view<T, IdxT> global_distances)
{
  size_t num_rows = static_cast<size_t>(global_neighbors.extent(0));
  size_t k        = static_cast<size_t>(global_neighbors.extent(1));

  bool select_min = cuvs::distance::is_min_close(metric);
  IdxT global_neighbors_fill_value =
    select_min ? std::numeric_limits<IdxT>::max() : std::numeric_limits<IdxT>::min();
  T global_distances_fill_value =
    select_min ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();

  std::fill(global_neighbors.data_handle(),
            global_neighbors.data_handle() + num_rows * k,
            global_neighbors_fill_value);
  std::fill(global_distances.data_handle(),
            global_distances.data_handle() + num_rows * k,
            global_distances_fill_value);
}

/**
 * Run balanced kmeans on a subsample of the dataset to get centroids.
 * Arguments:
 * - [in] res: raft resource
 * - [in] metric
 * - [in] dataset [num_rows x num_cols]: entire dataset on host
 * - [out] centroids [n_clusters x num_cols] : centroid vectors
 */
template <typename T, typename IdxT>
void get_centroids_on_data_subsample(raft::resources const& res,
                                     cuvs::distance::DistanceType metric,
                                     raft::host_matrix_view<const T, IdxT, row_major> dataset,
                                     raft::device_matrix_view<T, IdxT> centroids)
{
  size_t num_rows       = static_cast<size_t>(dataset.extent(0));
  size_t num_cols       = static_cast<size_t>(dataset.extent(1));
  size_t n_clusters     = centroids.extent(0);
  size_t num_subsamples = std::min(static_cast<size_t>(num_rows / n_clusters), 50000lu);

  if (num_subsamples <= 1000) {
    // heuristically running kmeans for something this small doesn't work well
    num_subsamples = std::min(num_rows, 5000lu);
  }
  auto dataset_subsample_d = raft::make_device_matrix<T, IdxT>(res, num_subsamples, num_cols);

  raft::matrix::sample_rows<T, IdxT>(
    res, raft::random::RngState{0}, dataset, dataset_subsample_d.view());
  cuvs::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.metric = metric;

  cuvs::cluster::kmeans::fit(
    res, kmeans_params, raft::make_const_mdspan(dataset_subsample_d.view()), centroids);
}

template <typename T, typename IdxT>
void single_gpu_assign_clusters(
  raft::resources const& res,
  size_t overlap_factor,
  size_t n_clusters,
  size_t n_rows_per_batch,
  size_t base_row_offset,
  raft::host_matrix_view<const T, IdxT, row_major> dataset,
  raft::device_matrix_view<T, IdxT, raft::row_major> centroids,
  cuvs::distance::DistanceType metric,
  raft::host_matrix_view<IdxT, IdxT, raft::row_major> global_nearest_cluster)
{
  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t num_cols = static_cast<size_t>(dataset.extent(1));

  // number of batches for this GPU to process
  size_t num_batches = n_clusters;

  auto dataset_batch_d =
    raft::make_device_matrix<T, IdxT, raft::row_major>(res, n_rows_per_batch, num_cols);

  auto nearest_clusters_idx_d =
    raft::make_device_matrix<IdxT, int64_t, raft::row_major>(res, n_rows_per_batch, overlap_factor);
  auto nearest_clusters_dist_d =
    raft::make_device_matrix<T, int64_t, raft::row_major>(res, n_rows_per_batch, overlap_factor);

  std::optional<raft::device_vector_view<const T, int64_t>> norms_view;
  cuvs::neighbors::brute_force::index<T> brute_force_index(res, centroids, norms_view, metric);

  for (size_t i = 0; i < num_batches; i++) {
    size_t row_offset              = n_rows_per_batch * i + base_row_offset;
    size_t n_rows_of_current_batch = std::min(n_rows_per_batch, num_rows - row_offset);
    raft::copy(dataset_batch_d.data_handle(),
               dataset.data_handle() + row_offset * num_cols,
               n_rows_of_current_batch * num_cols,
               resource::get_cuda_stream(res));

    // n_clusters is usually not large, so okay to do this brute-force
    cuvs::neighbors::brute_force::search(res,
                                         brute_force_index,
                                         raft::make_const_mdspan(dataset_batch_d.view()),
                                         nearest_clusters_idx_d.view(),
                                         nearest_clusters_dist_d.view());
    raft::copy(global_nearest_cluster.data_handle() + row_offset * overlap_factor,
               nearest_clusters_idx_d.data_handle(),
               n_rows_of_current_batch * overlap_factor,
               resource::get_cuda_stream(res));
  }
}

/**
 * Assign each data point to top overlap_factor number of clusters. Loads the data in batches
 * onto device for efficiency. Arguments:
 * - [in] res: raft resource
 * - [in] params: params for graph building
 * - [in] dataset [num_rows x num_cols]: entire dataset located on host memory
 * - [in] centroids [n_clusters x num_cols] : centroid vectors
 * - [out] global_nearest_cluster [num_rows X overlap_factor] : top overlap_factor closest
 * clusters for each data point
 */
template <typename T, typename IdxT>
void assign_clusters(raft::resources const& res,
                     const all_neighbors_params& params,
                     raft::host_matrix_view<const T, IdxT, row_major> dataset,
                     raft::device_matrix_view<T, IdxT, raft::row_major> centroids,
                     raft::host_matrix_view<IdxT, IdxT, raft::row_major> global_nearest_cluster)
{
  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t num_cols = static_cast<size_t>(dataset.extent(1));

  auto centroids_h = raft::make_host_matrix<T, IdxT>(params.n_clusters, num_cols);
  raft::copy(centroids_h.data_handle(),
             centroids.data_handle(),
             params.n_clusters * num_cols,
             raft::resource::get_cuda_stream(res));

  size_t n_rows_per_cluster = (num_rows + params.n_clusters - 1) / params.n_clusters;

  if (raft::resource::is_multi_gpu(res)) {
    int num_ranks = raft::resource::get_num_ranks(res);
    // multi-gpu assign clusters
    size_t clusters_per_rank = params.n_clusters / num_ranks;
    size_t rem               = params.n_clusters - clusters_per_rank * num_ranks;

#pragma omp parallel for num_threads(num_ranks)
    for (int rank = 0; rank < num_ranks; rank++) {
      auto dev_res = raft::resource::set_current_device_to_rank(res, rank);

      auto centroids_matrix = raft::make_device_matrix<T, IdxT>(res, params.n_clusters, num_cols);
      raft::copy(centroids_matrix.data_handle(),
                 centroids_h.data_handle(),
                 params.n_clusters * num_cols,
                 raft::resource::get_cuda_stream(dev_res));

      size_t base_cluster_idx = rank * clusters_per_rank + std::min((size_t)rank, rem);

      size_t n_clusters_for_this_rank =
        (size_t)rank < rem ? clusters_per_rank + 1 : clusters_per_rank;
      size_t base_row_offset_for_this_rank = n_rows_per_cluster * base_cluster_idx;

      single_gpu_assign_clusters(dev_res,
                                 params.overlap_factor,
                                 n_clusters_for_this_rank,
                                 n_rows_per_cluster,
                                 base_row_offset_for_this_rank,
                                 dataset,
                                 centroids_matrix.view(),
                                 params.metric,
                                 global_nearest_cluster);
    }
  } else {
    single_gpu_assign_clusters(res,
                               params.overlap_factor,
                               params.n_clusters,
                               n_rows_per_cluster,
                               0,
                               dataset,
                               centroids,
                               params.metric,
                               global_nearest_cluster);
  }
}

/**
 * Getting data indices that belong to cluster
 * Arguments:
 * - [in] res: raft resource
 * - [in] global_nearest_cluster [num_rows X overlap_factor] : top overlap_factor closest
 * clusters for each data point
 * - [out] inverted_indices [num_rows x overlap_factor sized vector] : vector for data indices
 * for each cluster
 * - [out] cluster_sizes [n_cluster] : cluster size for each cluster
 * - [out] cluster_offsets [n_cluster] : offset in inverted_indices for each cluster
 */
template <typename IdxT = int64_t>
void get_inverted_indices(raft::resources const& res,
                          raft::host_matrix_view<IdxT, IdxT> global_nearest_cluster,
                          raft::host_vector_view<IdxT, IdxT> inverted_indices,
                          raft::host_vector_view<IdxT, IdxT> cluster_sizes,
                          raft::host_vector_view<IdxT, IdxT> cluster_offsets)
{
  // build sparse inverted indices and get number of data points for each cluster
  size_t num_rows       = global_nearest_cluster.extent(0);
  size_t overlap_factor = global_nearest_cluster.extent(1);
  size_t n_clusters     = cluster_sizes.extent(0);

  auto local_offsets = raft::make_host_vector<IdxT>(n_clusters);

  std::fill(cluster_sizes.data_handle(), cluster_sizes.data_handle() + n_clusters, 0);
  std::fill(local_offsets.data_handle(), local_offsets.data_handle() + n_clusters, 0);

  for (size_t i = 0; i < num_rows; i++) {
    for (size_t j = 0; j < overlap_factor; j++) {
      IdxT cluster_id = global_nearest_cluster(i, j);
      cluster_sizes(cluster_id) += 1;
    }
  }

  cluster_offsets(0) = 0;
  for (size_t i = 1; i < n_clusters; i++) {
    cluster_offsets(i) = cluster_offsets(i - 1) + cluster_sizes(i - 1);
  }
  for (size_t i = 0; i < num_rows; i++) {
    for (size_t j = 0; j < overlap_factor; j++) {
      IdxT cluster_id = global_nearest_cluster(i, j);
      inverted_indices(cluster_offsets(cluster_id) + local_offsets(cluster_id)) = i;
      local_offsets(cluster_id) += 1;
    }
  }
}

template <typename IdxT = int64_t>
void get_min_max_cluster_size(size_t k,
                              size_t& max_cluster_size,
                              size_t& min_cluster_size,
                              raft::host_vector_view<IdxT, IdxT> cluster_sizes)
{
  max_cluster_size = 0;
  min_cluster_size = std::numeric_limits<size_t>::max();

  size_t n_clusters = cluster_sizes.extent(0);
  max_cluster_size  = static_cast<size_t>(
    *std::max_element(cluster_sizes.data_handle(), cluster_sizes.data_handle() + n_clusters));
  min_cluster_size = static_cast<size_t>(*std::min_element(
    cluster_sizes.data_handle(), cluster_sizes.data_handle() + n_clusters, [k](size_t a, size_t b) {
      return (a > k && (b <= k || a < b));  // Compare only elements larger than k
    }));
}

template <typename T, typename IdxT>
void single_gpu_batch_build(const raft::resources& handle,
                            raft::host_matrix_view<const T, IdxT, row_major> dataset,
                            detail::all_neighbors_builder<T, IdxT>& knn_builder,
                            size_t n_clusters,
                            raft::managed_matrix_view<IdxT, IdxT> global_neighbors,
                            raft::managed_matrix_view<T, IdxT> global_distances,
                            raft::host_vector_view<IdxT, IdxT, row_major> cluster_sizes,
                            raft::host_vector_view<IdxT, IdxT, row_major> cluster_offsets,
                            raft::host_vector_view<IdxT, IdxT, row_major> inverted_indices)
{
  size_t num_rows = dataset.extent(0);
  size_t num_cols = dataset.extent(1);

  auto cluster_data =
    raft::make_host_matrix<T, IdxT, row_major>(knn_builder.max_cluster_size, num_cols);

  knn_builder.prepare_build(dataset);

  for (size_t cluster_id = 0; cluster_id < n_clusters; cluster_id++) {
    size_t num_data_in_cluster = cluster_sizes(cluster_id);
    size_t offset              = cluster_offsets(cluster_id);
    std::cout << "[THREAD " << omp_get_thread_num() << "] cluster " << cluster_id + 1 << " / "
              << n_clusters << "(" << num_data_in_cluster << ")" << std::endl;
    if (num_data_in_cluster < knn_builder.k) {
      // for the unlikely event where clustering was done lopsidedly and this cluster has less than
      // k data
      continue;
    }

    // gathering dataset from host to host memory
    // cluster_data is on host for now because NN Descent allocates a new device matrix for data of
    // half fp regardless of whether data is on host or device
#pragma omp parallel for
    for (size_t i = 0; i < num_data_in_cluster; i++) {
      for (size_t j = 0; j < num_cols; j++) {
        size_t global_row  = inverted_indices(offset + i);
        cluster_data(i, j) = dataset(global_row, j);
      }
    }

    auto cluster_data_view = raft::make_host_matrix_view<const T, IdxT>(
      cluster_data.data_handle(), num_data_in_cluster, num_cols);
    auto inverted_indices_view = raft::make_host_vector_view<IdxT, IdxT>(
      inverted_indices.data_handle() + offset, num_data_in_cluster);

    knn_builder.build_knn(
      cluster_data_view, inverted_indices_view, global_neighbors, global_distances);
  }
}

template <typename T, typename IdxT>
void multi_gpu_batch_build(
  const raft::resources& handle,
  const all_neighbors_params& params,
  raft::host_matrix_view<const T, IdxT, row_major> dataset,
  raft::managed_matrix_view<IdxT, IdxT> global_neighbors,
  raft::managed_matrix_view<T, IdxT> global_distances,
  raft::host_vector_view<IdxT, IdxT, row_major> cluster_sizes,
  raft::host_vector_view<IdxT, IdxT, row_major> cluster_offsets,
  raft::host_vector_view<IdxT, IdxT, row_major> inverted_indices,
  std::optional<raft::device_vector_view<T, IdxT, row_major>> core_distances = std::nullopt,
  T alpha                                                                    = 1.0)
{
  size_t num_rows = dataset.extent(0);
  size_t num_cols = dataset.extent(1);
  size_t k        = global_neighbors.extent(1);

  int num_ranks = raft::resource::get_num_ranks(handle);

  size_t clusters_per_rank = params.n_clusters / num_ranks;
  size_t rem               = params.n_clusters - clusters_per_rank * num_ranks;

  std::vector<size_t> rank_offsets(num_ranks);
  std::vector<size_t> num_data_per_rank(num_ranks);
  std::vector<size_t> num_clusters_per_rank(num_ranks);
  std::vector<size_t> base_cluster_indices(num_ranks);

#pragma omp parallel for num_threads(num_ranks)
  for (int rank = 0; rank < num_ranks; rank++) {
    auto dev_res = raft::resource::set_current_device_to_rank(handle, rank);

    // This part is to distribute clusters across ranks as equally as possible
    // E.g. if we had 5 clusters and 3 ranks, instead of splitting into 1, 1, 3
    // we split into 2, 2, 1
    size_t num_data_for_this_rank = 0;
    size_t base_cluster_idx       = rank * clusters_per_rank + std::min((size_t)rank, rem);
    size_t num_clusters_for_this_rank =
      (size_t)rank < rem ? clusters_per_rank + 1 : clusters_per_rank;
    for (size_t p = 0; p < num_clusters_for_this_rank; p++) {
      num_data_for_this_rank += cluster_sizes(base_cluster_idx + p);
    }

    size_t rank_offset = cluster_offsets(base_cluster_idx);

    rank_offsets[rank]          = rank_offset;
    num_data_per_rank[rank]     = num_data_for_this_rank;
    num_clusters_per_rank[rank] = num_clusters_for_this_rank;
    base_cluster_indices[rank]  = base_cluster_idx;

    // remap offsets for each rank
    for (size_t p = 0; p < num_clusters_for_this_rank; p++) {
      cluster_offsets(base_cluster_idx + p) -= rank_offset;
    }

    auto cluster_sizes_for_this_rank = raft::make_host_vector_view<IdxT, IdxT>(
      cluster_sizes.data_handle() + base_cluster_idx, num_clusters_for_this_rank);
    auto cluster_offsets_for_this_rank = raft::make_host_vector_view<IdxT, IdxT>(
      cluster_offsets.data_handle() + base_cluster_idx, num_clusters_for_this_rank);
    auto inverted_indices_for_this_rank = raft::make_host_vector_view<IdxT, IdxT>(
      inverted_indices.data_handle() + rank_offset, num_data_for_this_rank);

    size_t max_cluster_size, min_cluster_size;
    get_min_max_cluster_size(k, max_cluster_size, min_cluster_size, cluster_sizes_for_this_rank);

    std::unique_ptr<all_neighbors_builder<T, IdxT>> knn_builder =
      get_knn_builder<T, IdxT>(dev_res, params, min_cluster_size, max_cluster_size, k);

    single_gpu_batch_build(dev_res,
                           dataset,
                           *knn_builder,
                           num_clusters_for_this_rank,
                           global_neighbors,
                           global_distances,
                           cluster_sizes_for_this_rank,
                           cluster_offsets_for_this_rank,
                           inverted_indices_for_this_rank);
  }

  if (core_distances.has_value()) {
    raft::matrix::shift(
      handle,
      raft::make_device_matrix_view<T, IdxT>(global_distances.data_handle(), num_rows, k),
      1,
      std::make_optional(static_cast<T>(0.0)));
    cuvs::neighbors::detail::reachability::core_distances<IdxT, T>(
      global_distances.data_handle(),
      k,
      k,
      num_rows,
      core_distances.value().data_handle(),
      raft::resource::get_cuda_stream(handle));

    // copy to host
    auto core_distances_h = raft::make_host_vector<T, IdxT>(num_rows);
    raft::copy(core_distances_h.data_handle(),
               core_distances.value().data_handle(),
               num_rows,
               raft::resource::get_cuda_stream(handle));

    reset_global_matrices(handle, params.metric, global_neighbors, global_distances);

#pragma omp parallel for num_threads(num_ranks)
    for (int rank = 0; rank < num_ranks; rank++) {
      auto dev_res                   = raft::resource::set_current_device_to_rank(handle, rank);
      auto core_distances_d_for_rank = raft::make_device_vector<T, IdxT>(dev_res, num_rows);

      raft::copy(core_distances_d_for_rank.data_handle(),
                 core_distances_h.data_handle(),
                 num_rows,
                 raft::resource::get_cuda_stream(dev_res));

      if (params.metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
        // comparison within nn descent for L2SqrtExpanded is done without applying sqrt.
        raft::linalg::map(dev_res,
                          core_distances_d_for_rank.view(),
                          raft::sq_op{},
                          raft::make_const_mdspan(core_distances_d_for_rank.view()));
      }

      auto dist_epilogue = cuvs::neighbors::detail::reachability::ReachabilityPostProcess<int, T>{
        core_distances_d_for_rank.data_handle(), alpha, num_rows};

      auto cluster_sizes_for_this_rank = raft::make_host_vector_view<IdxT, IdxT>(
        cluster_sizes.data_handle() + base_cluster_indices[rank], num_clusters_per_rank[rank]);
      auto cluster_offsets_for_this_rank = raft::make_host_vector_view<IdxT, IdxT>(
        cluster_offsets.data_handle() + base_cluster_indices[rank], num_clusters_per_rank[rank]);
      auto inverted_indices_for_this_rank = raft::make_host_vector_view<IdxT, IdxT>(
        inverted_indices.data_handle() + rank_offsets[rank], num_data_per_rank[rank]);

      size_t max_cluster_size, min_cluster_size;
      get_min_max_cluster_size(k, max_cluster_size, min_cluster_size, cluster_sizes_for_this_rank);

      auto knn_builder =
        get_knn_builder<T,
                        IdxT,
                        cuvs::neighbors::detail::reachability::ReachabilityPostProcess<int, T>>(
          dev_res,
          params,
          min_cluster_size,
          max_cluster_size,
          k,
          std::nullopt,
          std::nullopt,
          dist_epilogue);
      single_gpu_batch_build(dev_res,
                             dataset,
                             *knn_builder,
                             num_clusters_per_rank[rank],
                             global_neighbors,
                             global_distances,
                             cluster_sizes_for_this_rank,
                             cluster_offsets_for_this_rank,
                             inverted_indices_for_this_rank);
    }

    raft::matrix::shift(
      handle,
      raft::make_device_matrix_view<IdxT, IdxT>(global_neighbors.data_handle(), num_rows, k),
      1);
    raft::matrix::shift(
      handle,
      raft::make_device_matrix_view<T, IdxT>(global_distances.data_handle(), num_rows, k),
      raft::make_device_matrix_view<const T, IdxT>(
        core_distances.value().data_handle(), num_rows, 1));
  }
}

template <typename T, typename IdxT>
void batch_build(
  const raft::resources& handle,
  const all_neighbors_params& params,
  raft::host_matrix_view<const T, IdxT, row_major> dataset,
  raft::device_matrix_view<IdxT, IdxT, row_major> indices,
  std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances      = std::nullopt,
  std::optional<raft::device_vector_view<T, IdxT, row_major>> core_distances = std::nullopt,
  T alpha                                                                    = 1.0)
{
  if (raft::resource::is_multi_gpu(handle)) {
    RAFT_EXPECTS(params.n_clusters >= raft::resource::get_num_ranks(handle),
                 "n_clusters must be larger than or equal to number of GPUs for multi gpu "
                 "all-neighbors build");
    // For efficient CPU-computation of omp parallel for regions per GPU
    omp_set_nested(1);
  }

  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t num_cols = static_cast<size_t>(dataset.extent(1));
  size_t k        = indices.extent(1);

  RAFT_EXPECTS(params.n_clusters > params.overlap_factor,
               "overlap_factor should be smaller than n_clusters. We recommend starting from "
               "overlap_factor=2 and gradually increase it for better knn graph recall.");

  auto centroids = raft::make_device_matrix<T, IdxT>(handle, params.n_clusters, num_cols);
  get_centroids_on_data_subsample<T, IdxT>(handle, params.metric, dataset, centroids.view());

  auto global_nearest_cluster = raft::make_host_matrix<IdxT, IdxT>(num_rows, params.overlap_factor);
  assign_clusters<T, IdxT>(
    handle, params, dataset, centroids.view(), global_nearest_cluster.view());

  auto inverted_indices =
    raft::make_host_vector<IdxT, IdxT, raft::row_major>(num_rows * params.overlap_factor);
  auto cluster_sizes   = raft::make_host_vector<IdxT, IdxT, raft::row_major>(params.n_clusters);
  auto cluster_offsets = raft::make_host_vector<IdxT, IdxT, raft::row_major>(params.n_clusters);
  get_inverted_indices(handle,
                       global_nearest_cluster.view(),
                       inverted_indices.view(),
                       cluster_sizes.view(),
                       cluster_offsets.view());

  auto global_neighbors = raft::make_managed_matrix<IdxT, IdxT>(handle, num_rows, k);
  auto global_distances = raft::make_managed_matrix<T, IdxT>(handle, num_rows, k);

  reset_global_matrices(handle, params.metric, global_neighbors.view(), global_distances.view());

  if (raft::resource::is_multi_gpu(handle)) {
    multi_gpu_batch_build(handle,
                          params,
                          dataset,
                          global_neighbors.view(),
                          global_distances.view(),
                          cluster_sizes.view(),
                          cluster_offsets.view(),
                          inverted_indices.view(),
                          core_distances,
                          alpha);
  } else {
    size_t max_cluster_size, min_cluster_size;
    get_min_max_cluster_size(k, max_cluster_size, min_cluster_size, cluster_sizes.view());
    std::unique_ptr<all_neighbors_builder<T, IdxT>> knn_builder =
      get_knn_builder<T, IdxT>(handle, params, min_cluster_size, max_cluster_size, k);
    single_gpu_batch_build(handle,
                           dataset,
                           *knn_builder,
                           params.n_clusters,
                           global_neighbors.view(),
                           global_distances.view(),
                           cluster_sizes.view(),
                           cluster_offsets.view(),
                           inverted_indices.view());
    if (core_distances.has_value()) {
      size_t k = static_cast<size_t>(indices.extent(1));

      raft::matrix::shift(
        handle,
        raft::make_device_matrix_view<T, IdxT>(global_distances.data_handle(), num_rows, k),
        1,
        std::make_optional(static_cast<T>(0.0)));
      cuvs::neighbors::detail::reachability::core_distances<IdxT, T>(
        global_distances.data_handle(),
        k,
        k,
        num_rows,
        core_distances.value().data_handle(),
        raft::resource::get_cuda_stream(handle));

      reset_global_matrices(
        handle, params.metric, global_neighbors.view(), global_distances.view());

      if (params.metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
        // comparison within nn descent for L2SqrtExpanded is done without applying sqrt.
        raft::linalg::map(handle,
                          core_distances.value(),
                          raft::sq_op{},
                          raft::make_const_mdspan(core_distances.value()));
      }

      auto dist_epilogue = cuvs::neighbors::detail::reachability::ReachabilityPostProcess<int, T>{
        core_distances.value().data_handle(), alpha, num_rows};

      auto knn_builder =
        get_knn_builder<T,
                        IdxT,
                        cuvs::neighbors::detail::reachability::ReachabilityPostProcess<int, T>>(
          handle,
          params,
          min_cluster_size,
          max_cluster_size,
          k,
          std::nullopt,
          std::nullopt,
          dist_epilogue);
      single_gpu_batch_build(handle,
                             dataset,
                             *knn_builder,
                             params.n_clusters,
                             global_neighbors.view(),
                             global_distances.view(),
                             cluster_sizes.view(),
                             cluster_offsets.view(),
                             inverted_indices.view());

      raft::matrix::shift(
        handle,
        raft::make_device_matrix_view<IdxT, IdxT>(global_neighbors.data_handle(), num_rows, k),
        1);
      raft::matrix::shift(
        handle,
        raft::make_device_matrix_view<T, IdxT>(global_distances.data_handle(), num_rows, k),
        raft::make_device_matrix_view<const T, IdxT>(
          core_distances.value().data_handle(), num_rows, 1));
    }
  }

  raft::copy(indices.data_handle(),
             global_neighbors.data_handle(),
             num_rows * k,
             raft::resource::get_cuda_stream(handle));
  if (distances.has_value()) {
    raft::copy(distances.value().data_handle(),
               global_distances.data_handle(),
               num_rows * k,
               raft::resource::get_cuda_stream(handle));
  }
}

}  // namespace cuvs::neighbors::all_neighbors::detail
