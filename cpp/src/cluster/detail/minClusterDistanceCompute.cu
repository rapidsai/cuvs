/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans_common.cuh"

namespace cuvs::cluster::kmeans::detail {

// Calculates a <key, value> pair for every sample in input 'X' where key is an
// index to an sample in 'centroids' (index of the nearest centroid) and 'value'
// is the distance between the sample and the 'centroid[key]'
template <typename DataT, typename IndexT>
void minClusterAndDistanceCompute(  // NOLINT(readability-identifier-naming)
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, IndexT> X,
  raft::device_matrix_view<const DataT, IndexT> centroids,
  raft::device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT> minClusterAndDistance,
  raft::device_vector_view<const DataT, IndexT> L2NormX,
  rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
  cuvs::distance::DistanceType metric,
  int batch_samples,
  int batch_centroids,
  rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = centroids.extent(0);
  // todo(lsugy): change batch size computation when using fusedL2NN!
  bool is_fused = metric == cuvs::distance::DistanceType::L2Expanded ||
                  metric == cuvs::distance::DistanceType::L2SqrtExpanded;
  auto data_batch_size =
    is_fused ? static_cast<IndexT>(n_samples) : getDataBatchSize(batch_samples, n_samples);
  auto centroids_batch_size = getCentroidsBatchSize(batch_centroids, n_clusters);

  if (is_fused) {
    L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(L2NormBuf_OR_DistBuf.data(),
                                                      centroids.data_handle(),
                                                      centroids.extent(1),
                                                      centroids.extent(0),
                                                      stream);
  } else {
    // TODO(cuvs): Unless pool allocator is used, passing in a workspace for this  //
    // NOLINT(google-readability-todo) isn't really increasing performance because this needs to do
    // a re-allocation anyways. ref https://github.com/rapidsai/raft/issues/930
    L2NormBuf_OR_DistBuf.resize(data_batch_size * centroids_batch_size, stream);
  }

  // Note - pairwise_distance and centroids_norm share the same buffer
  // centroids_norm [n_clusters] - tensor wrapper around centroids L2 Norm
  auto centroids_norm =
    raft::make_device_vector_view<DataT, IndexT>(L2NormBuf_OR_DistBuf.data(), n_clusters);
  // pairwise_distance[ns x nc] - tensor wrapper around the distance buffer
  auto pairwise_distance = raft::make_device_matrix_view<DataT, IndexT>(
    L2NormBuf_OR_DistBuf.data(), data_batch_size, centroids_batch_size);

  raft::KeyValuePair<IndexT, DataT> initial_value(0, std::numeric_limits<DataT>::max());

  thrust::fill(raft::resource::get_thrust_policy(handle),
               minClusterAndDistance.data_handle(),
               minClusterAndDistance.data_handle() + minClusterAndDistance.size(),
               initial_value);

  // tile over the input dataset
  for (IndexT d_idx = 0; d_idx < n_samples; d_idx += data_batch_size) {
    // # of samples for the current batch
    auto ns = std::min(static_cast<IndexT>(data_batch_size), n_samples - d_idx);

    // dataset_view [ns x n_features] - view representing the current batch of
    // input dataset
    auto dataset_view = raft::make_device_matrix_view<const DataT, IndexT>(
      X.data_handle() + (d_idx * n_features), ns, n_features);

    // min_cluster_and_distance_view [ns x n_clusters]
    auto min_cluster_and_distance_view =
      raft::make_device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT>(
        minClusterAndDistance.data_handle() + d_idx, ns);

    auto l2_norm_x_view =
      raft::make_device_vector_view<const DataT, IndexT>(L2NormX.data_handle() + d_idx, ns);

    if (is_fused) {
      workspace.resize((sizeof(int)) * ns, stream);

      // todo(lsugy): remove c_idx
      cuvs::distance::
        fused_distance_nn_min_reduce<DataT, raft::KeyValuePair<IndexT, DataT>, IndexT>(
          min_cluster_and_distance_view.data_handle(),
          dataset_view.data_handle(),
          centroids.data_handle(),
          l2_norm_x_view.data_handle(),
          centroids_norm.data_handle(),
          ns,
          n_clusters,
          n_features,
          reinterpret_cast<void*>(workspace.data()),
          metric != cuvs::distance::DistanceType::L2Expanded,
          false,
          true,
          metric,
          0.0f,
          stream);
    } else {
      // tile over the centroids
      for (IndexT c_idx = 0; c_idx < n_clusters; c_idx += centroids_batch_size) {
        // # of centroids for the current batch
        auto nc = std::min(static_cast<IndexT>(centroids_batch_size), n_clusters - c_idx);

        // centroids_view [nc x n_features] - view representing the current batch
        // of centroids
        auto centroids_view = raft::make_device_matrix_view<const DataT, IndexT>(
          centroids.data_handle() + (c_idx * n_features), nc, n_features);

        // pairwise_distance_view [ns x nc] - view representing the pairwise
        // distance for current batch
        auto pairwise_distance_view =
          raft::make_device_matrix_view<DataT, IndexT>(pairwise_distance.data_handle(), ns, nc);

        // calculate pairwise distance between current tile of cluster centroids
        // and input dataset
        pairwise_distance_kmeans<DataT, IndexT>(
          handle, dataset_view, centroids_view, pairwise_distance_view, metric);

        // argmin reduction returning <index, value> pair
        // calculates the closest centroid and the distance to the closest
        // centroid
        raft::linalg::coalescedReduction(
          min_cluster_and_distance_view.data_handle(),
          pairwise_distance_view.data_handle(),
          pairwise_distance_view.extent(1),
          pairwise_distance_view.extent(0),
          initial_value,
          stream,
          true,
          [=] __device__(const DataT val, const IndexT i) -> raft::KeyValuePair<IndexT, DataT> {
            raft::KeyValuePair<IndexT, DataT> pair;
            pair.key   = c_idx + i;
            pair.value = val;
            return pair;
          },
          raft::argmin_op{},
          raft::identity_op{});
      }
    }
  }
}

#define INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(DataT, IndexT)                                    \
  template void minClusterAndDistanceCompute<DataT, IndexT>(                                   \
    raft::resources const& handle,                                                             \
    raft::device_matrix_view<const DataT, IndexT> X,                                           \
    raft::device_matrix_view<const DataT, IndexT> centroids,                                   \
    raft::device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT> minClusterAndDistance, \
    raft::device_vector_view<const DataT, IndexT> L2NormX,                                     \
    rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,                                          \
    cuvs::distance::DistanceType metric,                                                       \
    int batch_samples,                                                                         \
    int batch_centroids,                                                                       \
    rmm::device_uvector<char>& workspace);

INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(float, int64_t)
INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(double, int64_t)
INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(float, int)
INSTANTIATE_MIN_CLUSTER_AND_DISTANCE(double, int)

#undef INSTANTIATE_MIN_CLUSTER_AND_DISTANCE

template <typename DataT, typename IndexT>
void minClusterDistanceCompute(  // NOLINT(readability-identifier-naming)
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, IndexT> X,
  raft::device_matrix_view<DataT, IndexT> centroids,
  raft::device_vector_view<DataT, IndexT> minClusterDistance,
  raft::device_vector_view<DataT, IndexT> L2NormX,
  rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
  cuvs::distance::DistanceType metric,
  int batch_samples,
  int batch_centroids,
  rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = centroids.extent(0);

  bool is_fused = metric == cuvs::distance::DistanceType::L2Expanded ||
                  metric == cuvs::distance::DistanceType::L2SqrtExpanded;
  auto data_batch_size =
    is_fused ? static_cast<IndexT>(n_samples) : getDataBatchSize(batch_samples, n_samples);
  auto centroids_batch_size = getCentroidsBatchSize(batch_centroids, n_clusters);

  if (is_fused) {
    L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(L2NormBuf_OR_DistBuf.data(),
                                                      centroids.data_handle(),
                                                      centroids.extent(1),
                                                      centroids.extent(0),
                                                      stream);
  } else {
    L2NormBuf_OR_DistBuf.resize(data_batch_size * centroids_batch_size, stream);
  }

  // Note - pairwise_distance and centroids_norm share the same buffer
  // centroids_norm [n_clusters] - tensor wrapper around centroids L2 Norm
  auto centroids_norm =
    raft::make_device_vector_view<DataT, IndexT>(L2NormBuf_OR_DistBuf.data(), n_clusters);
  // pairwise_distance[ns x nc] - tensor wrapper around the distance buffer
  auto pairwise_distance = raft::make_device_matrix_view<DataT, IndexT>(
    L2NormBuf_OR_DistBuf.data(), data_batch_size, centroids_batch_size);

  thrust::fill(raft::resource::get_thrust_policy(handle),
               minClusterDistance.data_handle(),
               minClusterDistance.data_handle() + minClusterDistance.size(),
               std::numeric_limits<DataT>::max());

  // tile over the input data and calculate distance matrix [n_samples x
  // n_clusters]
  for (IndexT d_idx = 0; d_idx < n_samples; d_idx += data_batch_size) {
    // # of samples for the current batch
    auto ns = std::min(static_cast<IndexT>(data_batch_size), n_samples - d_idx);

    // dataset_view [ns x n_features] - view representing the current batch of
    // input dataset
    auto dataset_view = raft::make_device_matrix_view<const DataT, IndexT>(
      X.data_handle() + d_idx * n_features, ns, n_features);

    // min_cluster_distance_view [ns x n_clusters]
    auto min_cluster_distance_view =
      raft::make_device_vector_view<DataT, IndexT>(minClusterDistance.data_handle() + d_idx, ns);

    auto l2_norm_x_view =
      raft::make_device_vector_view<DataT, IndexT>(L2NormX.data_handle() + d_idx, ns);

    if (is_fused) {
      workspace.resize((sizeof(IndexT)) * ns, stream);

      cuvs::distance::fused_distance_nn_min_reduce<DataT, DataT, IndexT>(
        min_cluster_distance_view.data_handle(),
        dataset_view.data_handle(),
        centroids.data_handle(),
        l2_norm_x_view.data_handle(),
        centroids_norm.data_handle(),
        ns,
        n_clusters,
        n_features,
        reinterpret_cast<void*>(workspace.data()),
        metric != cuvs::distance::DistanceType::L2Expanded,
        false,
        true,
        metric,
        0.0f,
        stream);
    } else {
      // tile over the centroids
      for (IndexT c_idx = 0; c_idx < n_clusters; c_idx += centroids_batch_size) {
        // # of centroids for the current batch
        auto nc = std::min(static_cast<IndexT>(centroids_batch_size), n_clusters - c_idx);

        // centroids_view [nc x n_features] - view representing the current batch
        // of centroids
        auto centroids_view = raft::make_device_matrix_view<DataT, IndexT>(
          centroids.data_handle() + c_idx * n_features, nc, n_features);

        // pairwise_distance_view [ns x nc] - view representing the pairwise
        // distance for current batch
        auto pairwise_distance_view =
          raft::make_device_matrix_view<DataT, IndexT>(pairwise_distance.data_handle(), ns, nc);

        // calculate pairwise distance between current tile of cluster centroids
        // and input dataset
        pairwise_distance_kmeans<DataT, IndexT>(
          handle, dataset_view, centroids_view, pairwise_distance_view, metric);

        raft::linalg::coalescedReduction(min_cluster_distance_view.data_handle(),
                                         pairwise_distance_view.data_handle(),
                                         pairwise_distance_view.extent(1),
                                         pairwise_distance_view.extent(0),
                                         std::numeric_limits<DataT>::max(),
                                         stream,
                                         true,
                                         raft::identity_op{},
                                         raft::min_op{},
                                         raft::identity_op{});
      }
    }
  }
}

#define INSTANTIATE_MIN_CLUSTER_DISTANCE(DataT, IndexT)         \
  template void minClusterDistanceCompute<DataT, IndexT>(       \
    raft::resources const& handle,                              \
    raft::device_matrix_view<const DataT, IndexT> X,            \
    raft::device_matrix_view<DataT, IndexT> centroids,          \
    raft::device_vector_view<DataT, IndexT> minClusterDistance, \
    raft::device_vector_view<DataT, IndexT> L2NormX,            \
    rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,           \
    cuvs::distance::DistanceType metric,                        \
    int batch_samples,                                          \
    int batch_centroids,                                        \
    rmm::device_uvector<char>& workspace);

INSTANTIATE_MIN_CLUSTER_DISTANCE(float, int64_t)
INSTANTIATE_MIN_CLUSTER_DISTANCE(double, int64_t)
INSTANTIATE_MIN_CLUSTER_DISTANCE(float, int)
INSTANTIATE_MIN_CLUSTER_DISTANCE(double, int)

#undef INSTANTIATE_MIN_CLUSTER_DISTANCE

}  // namespace cuvs::cluster::kmeans::detail
