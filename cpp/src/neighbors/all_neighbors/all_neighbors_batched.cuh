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
#include "all_neighbors_builder.cuh"
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/all_neighbors.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/refine.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/managed_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/sample_rows.cuh>
#include <raft/util/cudart_utils.hpp>
#include <variant>

namespace cuvs::neighbors::all_neighbors::detail {
using namespace cuvs::neighbors;

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
  size_t n_nearest_clusters,
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

  auto nearest_clusters_idx_d = raft::make_device_matrix<IdxT, int64_t, raft::row_major>(
    res, n_rows_per_batch, n_nearest_clusters);
  auto nearest_clusters_dist_d = raft::make_device_matrix<T, int64_t, raft::row_major>(
    res, n_rows_per_batch, n_nearest_clusters);

  std::optional<raft::device_vector_view<const T, int64_t>> norms_view;
  cuvs::neighbors::brute_force::index<T> brute_force_index(res, centroids, norms_view, metric);

  for (size_t i = 0; i < num_batches; i++) {
    size_t row_offset              = n_rows_per_batch * i + base_row_offset;
    size_t n_rows_of_current_batch = std::min(n_rows_per_batch, num_rows - row_offset);
    std::cout << "[THREAD " << omp_get_thread_num() << "] Assigning cluster " << i + 1 << " / "
              << num_batches << " row offset " << row_offset << std::endl;
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
    raft::copy(global_nearest_cluster.data_handle() + row_offset * n_nearest_clusters,
               nearest_clusters_idx_d.data_handle(),
               n_rows_of_current_batch * n_nearest_clusters,
               resource::get_cuda_stream(res));
  }
}

/**
 * Assign each data point to top n_nearest_clusters number of clusters. Loads the data in batches
 * onto device for efficiency. Arguments:
 * - [in] res: raft resource
 * - [in] n_nearest_clusters: number of nearest clusters that each data is assigned to
 * - [in] n_clusters: total number of clusters
 * - [in] dataset [num_rows x num_cols]: entire dataset on host
 * - [in] centroids [n_clusters x num_cols] : centroid vectors
 * - [in] metric
 * - [out] global_nearest_cluster [num_rows X n_nearest_clusters] : top n_nearest_clusters closest
 * clusters for each data point
 */
template <typename T, typename IdxT>
void assign_clusters(raft::resources const& res,
                     size_t n_nearest_clusters,
                     size_t n_clusters,
                     raft::host_matrix_view<const T, IdxT, row_major> dataset,
                     raft::device_matrix_view<T, IdxT, raft::row_major> centroids,
                     cuvs::distance::DistanceType metric,
                     raft::host_matrix_view<IdxT, IdxT, raft::row_major> global_nearest_cluster)
{
  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t num_cols = static_cast<size_t>(dataset.extent(1));
  // // auto centroids_view = raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
  // //   centroids.data_handle(), n_clusters, num_cols);
  // size_t num_batches      = n_clusters;
  // size_t n_rows_per_batch = (num_rows + n_clusters) / n_clusters;

  int num_ranks = 0;
  cudaGetDeviceCount(&num_ranks);

  auto centroids_h = raft::make_host_matrix<T, IdxT>(n_clusters, num_cols);
  raft::copy(centroids_h.data_handle(),
             centroids.data_handle(),
             n_clusters * num_cols,
             raft::resource::get_cuda_stream(res));

  if (num_ranks > 1) {
    // multi-gpu assign clusters
    size_t clusters_per_rank  = n_clusters / num_ranks;
    size_t rem                = n_clusters - clusters_per_rank * num_ranks;
    size_t n_rows_per_cluster = (num_rows + n_clusters - 1) / n_clusters;

#pragma omp parallel for num_threads(num_ranks)
    for (int rank = 0; rank < num_ranks; rank++) {
      auto dev_res = raft::device_resources{};
      RAFT_CUDA_TRY(cudaSetDevice(rank));

      auto centroids_matrix = raft::make_device_matrix<T, IdxT>(res, n_clusters, num_cols);
      raft::copy(centroids_matrix.data_handle(),
                 centroids_h.data_handle(),
                 n_clusters * num_cols,
                 raft::resource::get_cuda_stream(dev_res));

      size_t base_cluster_idx = rank * clusters_per_rank + std::min((size_t)rank, rem);

      size_t n_clusters_for_this_rank =
        (size_t)rank < rem ? clusters_per_rank + 1 : clusters_per_rank;
      size_t base_row_offset_for_this_rank = n_rows_per_cluster * base_cluster_idx;

      single_gpu_assign_clusters(dev_res,
                                 n_nearest_clusters,
                                 n_clusters_for_this_rank,
                                 n_rows_per_cluster,
                                 base_row_offset_for_this_rank,
                                 dataset,
                                 centroids_matrix.view(),
                                 metric,
                                 global_nearest_cluster);
    }
  } else {
    size_t n_rows_per_batch = (num_rows + n_clusters) / n_clusters;
    single_gpu_assign_clusters(res,
                               n_nearest_clusters,
                               n_clusters,
                               n_rows_per_batch,
                               0,
                               dataset,
                               centroids,
                               metric,
                               global_nearest_cluster);
  }
  // auto dataset_batch_d =
  //   raft::make_device_matrix<T, IdxT, raft::row_major>(res, n_rows_per_batch, num_cols);

  // auto nearest_clusters_idx_d = raft::make_device_matrix<IdxT, int64_t, raft::row_major>(
  //   res, n_rows_per_batch, num_nearest_clusters);
  // auto nearest_clusters_dist_d = raft::make_device_matrix<T, int64_t, raft::row_major>(
  //   res, n_rows_per_batch, num_nearest_clusters);

  // for (size_t i = 0; i < num_batches; i++) {
  //   std::cout << "Assigning cluster " << i+1 << " / " << num_batches << std::endl;
  //   size_t row_offset              = n_rows_per_batch * i;
  //   size_t n_rows_of_current_batch = std::min(n_rows_per_batch, num_rows - row_offset);
  //   raft::copy(dataset_batch_d.data_handle(),
  //              dataset.data_handle() + row_offset * num_cols,
  //              n_rows_of_current_batch * num_cols,
  //              resource::get_cuda_stream(res));
  //   std::optional<raft::device_vector_view<const T, int64_t>> norms_view;
  //   cuvs::neighbors::brute_force::index<T> brute_force_index(
  //     res, centroids.view(), norms_view, metric);

  //   // n_clusters is usually not large, so okay to do this brute-force
  //   cuvs::neighbors::brute_force::search(res,
  //                                        brute_force_index,
  //                                        raft::make_const_mdspan(dataset_batch_d.view()),
  //                                        nearest_clusters_idx_d.view(),
  //                                        nearest_clusters_dist_d.view());
  //   raft::copy(global_nearest_cluster.data_handle() + row_offset * num_nearest_clusters,
  //              nearest_clusters_idx_d.data_handle(),
  //              n_rows_of_current_batch * num_nearest_clusters,
  //              resource::get_cuda_stream(res));
  // }
}

/**
 * Some memory-heavy allocations that can be used over multiple clusters should be allocated here
 * Arguments:
 * - [in] res: raft resource
 * - [in] n_clusters: total number of clusters
 * - [in] k
 * - [out] max_cluster_size
 * - [out] min_cluster_size
 * - [in] global_nearest_cluster [num_rows X n_nearest_clusters] : top n_nearest_clusters closest
 * clusters for each data point
 * - [out] inverted_indices [num_rows x n_nearest_clusters sized vector] : vector for data indices
 * for each cluster
 * - [out] cluster_size [n_cluster] : cluster size for each cluster
 * - [out] cluster_offset [n_cluster] : offset in inverted_indices for each cluster
 */
template <typename IdxT = int64_t>
void get_inverted_indices(raft::resources const& res,
                          size_t n_clusters,
                          size_t k,
                          size_t& max_cluster_size,
                          size_t& min_cluster_size,
                          raft::host_matrix_view<IdxT, IdxT> global_nearest_cluster,
                          raft::host_vector_view<IdxT, IdxT> inverted_indices,
                          raft::host_vector_view<IdxT, IdxT> cluster_sizes,
                          raft::host_vector_view<IdxT, IdxT> cluster_offsets)
{
  // build sparse inverted indices and get number of data points for each cluster
  size_t num_rows           = global_nearest_cluster.extent(0);
  size_t n_nearest_clusters = global_nearest_cluster.extent(1);

  auto local_offsets = raft::make_host_vector<IdxT>(n_clusters);

  max_cluster_size = 0;
  min_cluster_size = std::numeric_limits<size_t>::max();

  std::fill(cluster_sizes.data_handle(), cluster_sizes.data_handle() + n_clusters, 0);
  std::fill(local_offsets.data_handle(), local_offsets.data_handle() + n_clusters, 0);

  for (size_t i = 0; i < num_rows; i++) {
    for (size_t j = 0; j < n_nearest_clusters; j++) {
      IdxT cluster_id = global_nearest_cluster(i, j);
      cluster_sizes(cluster_id) += 1;
    }
  }

  cluster_offsets(0) = 0;
  for (size_t i = 1; i < n_clusters; i++) {
    cluster_offsets(i) = cluster_offsets(i - 1) + cluster_sizes(i - 1);
  }
  for (size_t i = 0; i < num_rows; i++) {
    for (size_t j = 0; j < n_nearest_clusters; j++) {
      IdxT cluster_id = global_nearest_cluster(i, j);
      inverted_indices(cluster_offsets(cluster_id) + local_offsets(cluster_id)) = i;
      local_offsets(cluster_id) += 1;
    }
  }

  max_cluster_size = static_cast<size_t>(
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
                            raft::managed_matrix_view<IdxT, IdxT> global_neighbors,
                            raft::managed_matrix_view<T, IdxT> global_distances,
                            size_t max_cluster_size,
                            size_t n_clusters,
                            all_neighbors::index<IdxT, T>& index,
                            const index_params& batch_params,
                            raft::host_vector_view<IdxT, IdxT, row_major> cluster_sizes,
                            raft::host_vector_view<IdxT, IdxT, row_major> cluster_offsets,
                            raft::host_vector_view<IdxT, IdxT, row_major> inverted_indices)
{
  size_t num_rows = dataset.extent(0);
  size_t num_cols = dataset.extent(1);

  auto cluster_data = raft::make_host_matrix<T, IdxT, row_major>(max_cluster_size, num_cols);

  knn_builder.prepare_build(dataset);

  for (size_t cluster_id = 0; cluster_id < n_clusters; cluster_id++) {
    size_t num_data_in_cluster = cluster_sizes(cluster_id);
    std::cout << "[THREAD " << omp_get_thread_num() << "] cluster " << cluster_id + 1 << " / "
              << n_clusters << "(" << num_data_in_cluster << ")" << std::endl;
    size_t offset = cluster_offsets(cluster_id);
    if (num_data_in_cluster < static_cast<size_t>(index.k())) {
      // for the unlikely event where clustering was done lopsidedly and this cluster has less than
      // k data
      continue;
    }

    // gathering dataset from host
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

    knn_builder.build_knn(batch_params,
                          cluster_data_view,
                          index,
                          inverted_indices_view,
                          global_neighbors,
                          global_distances);
  }
}

template <typename T, typename IdxT>
void multi_gpu_batch_build(const raft::resources& handle,
                           raft::host_matrix_view<const T, IdxT, row_major> dataset,
                           raft::managed_matrix_view<IdxT, IdxT> global_neighbors,
                           raft::managed_matrix_view<T, IdxT> global_distances,
                           size_t max_cluster_size,
                           size_t min_cluster_size,
                           all_neighbors::index<IdxT, T>& index,
                           const index_params& params,
                           raft::host_vector_view<IdxT, IdxT, row_major> cluster_sizes,
                           raft::host_vector_view<IdxT, IdxT, row_major> cluster_offsets,
                           raft::host_vector_view<IdxT, IdxT, row_major> inverted_indices)
{
  size_t num_rows = dataset.extent(0);
  size_t num_cols = dataset.extent(1);

  int num_ranks = 0;
  cudaGetDeviceCount(&num_ranks);

  size_t clusters_per_rank = params.n_clusters / num_ranks;
  size_t rem               = params.n_clusters - clusters_per_rank * num_ranks;

#pragma omp parallel for num_threads(num_ranks)
  for (int rank = 0; rank < num_ranks; rank++) {
    auto dev_res = raft::device_resources{};
    RAFT_CUDA_TRY(cudaSetDevice(rank));

    std::unique_ptr<all_neighbors_builder<T, IdxT>> knn_builder = get_knn_builder<T, IdxT>(
      dev_res, params, static_cast<size_t>(index.k()), min_cluster_size, max_cluster_size, true);

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
    single_gpu_batch_build(dev_res,
                           dataset,
                           *knn_builder,
                           global_neighbors,
                           global_distances,
                           max_cluster_size,
                           num_clusters_for_this_rank,
                           index,
                           params,
                           cluster_sizes_for_this_rank,
                           cluster_offsets_for_this_rank,
                           inverted_indices_for_this_rank);
  }
}

template <typename T, typename IdxT>
void batch_build(const raft::resources& handle,
                 raft::host_matrix_view<const T, IdxT, row_major> dataset,
                 const index_params& batch_params,
                 all_neighbors::index<IdxT, T>& index)
{
  int num_ranks = 0;
  cudaGetDeviceCount(&num_ranks);
  if (num_ranks > 1) {
    // For efficient CPU-computation of omp parallel for regions per GPU
    omp_set_nested(1);
  }

  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t num_cols = static_cast<size_t>(dataset.extent(1));

  size_t n_nearest_clusters = batch_params.n_nearest_clusters;
  size_t n_clusters         = batch_params.n_clusters;
  RAFT_EXPECTS(n_clusters > n_nearest_clusters,
               "n_nearest_clusters should be smaller than n_clusters. Recommend starting from "
               "n_nearest_clusters=2 and gradually increase it for better knn graph recall.");

  auto start     = raft::curTimeMillis();
  auto centroids = raft::make_device_matrix<T, IdxT>(handle, n_clusters, num_cols);
  get_centroids_on_data_subsample<T, IdxT>(handle, batch_params.metric, dataset, centroids.view());
  auto end = raft::curTimeMillis();

  std::cout << "done getting centroids on data subsample " << end - start << std::endl;
  ;

  auto global_nearest_cluster = raft::make_host_matrix<IdxT, IdxT>(num_rows, n_nearest_clusters);
  start                       = raft::curTimeMillis();
  assign_clusters<T, IdxT>(handle,
                           n_nearest_clusters,
                           n_clusters,
                           dataset,
                           centroids.view(),
                           batch_params.metric,
                           global_nearest_cluster.view());
  end = raft::curTimeMillis();
  std::cout << "done assigning clusters " << end - start << std::endl;
  ;

  start = raft::curTimeMillis();
  auto inverted_indices =
    raft::make_host_vector<IdxT, IdxT, raft::row_major>(num_rows * n_nearest_clusters);
  auto cluster_sizes   = raft::make_host_vector<IdxT, IdxT, raft::row_major>(n_clusters);
  auto cluster_offsets = raft::make_host_vector<IdxT, IdxT, raft::row_major>(n_clusters);
  size_t max_cluster_size, min_cluster_size;
  get_inverted_indices(handle,
                       n_clusters,
                       index.k(),
                       max_cluster_size,
                       min_cluster_size,
                       global_nearest_cluster.view(),
                       inverted_indices.view(),
                       cluster_sizes.view(),
                       cluster_offsets.view());
  end = raft::curTimeMillis();
  std::cout << "done getting inverted indices " << end - start << std::endl;
  ;

  auto global_neighbors = raft::make_managed_matrix<IdxT, IdxT>(handle, num_rows, index.k());
  auto global_distances = raft::make_managed_matrix<float, IdxT>(handle, num_rows, index.k());
  std::fill(global_neighbors.data_handle(),
            global_neighbors.data_handle() + num_rows * index.k(),
            std::numeric_limits<IdxT>::max());
  std::fill(global_distances.data_handle(),
            global_distances.data_handle() + num_rows * index.k(),
            std::numeric_limits<float>::max());

  if (num_ranks > 1) {
    multi_gpu_batch_build(handle,
                          dataset,
                          global_neighbors.view(),
                          global_distances.view(),
                          max_cluster_size,
                          min_cluster_size,
                          index,
                          batch_params,
                          cluster_sizes.view(),
                          cluster_offsets.view(),
                          inverted_indices.view());
  } else {
    std::unique_ptr<all_neighbors_builder<T, IdxT>> knn_builder =
      get_knn_builder<T, IdxT>(handle,
                               batch_params,
                               static_cast<size_t>(index.k()),
                               min_cluster_size,
                               max_cluster_size,
                               true);
    single_gpu_batch_build(handle,
                           dataset,
                           *knn_builder,
                           global_neighbors.view(),
                           global_distances.view(),
                           max_cluster_size,
                           n_clusters,
                           index,
                           batch_params,
                           cluster_sizes.view(),
                           cluster_offsets.view(),
                           inverted_indices.view());
  }

  raft::copy(index.graph().data_handle(),
             global_neighbors.data_handle(),
             num_rows * index.k(),
             raft::resource::get_cuda_stream(handle));
  if (index.return_distances() && index.distances().has_value()) {
    raft::copy(index.distances().value().data_handle(),
               global_distances.data_handle(),
               num_rows * index.k(),
               raft::resource::get_cuda_stream(handle));
  }
}

}  // namespace cuvs::neighbors::all_neighbors::detail
