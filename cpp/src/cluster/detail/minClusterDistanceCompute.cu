/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../distance/fused_distance_nn.cuh"
#include "../../distance/unfused_distance_nn.cuh"
#include "kmeans_common.cuh"

#include <raft/matrix/init.cuh>

namespace cuvs::cluster::kmeans::detail {

namespace {

// Determinism self-test for the fused L2 nearest-neighbor reduction used by
// minClusterAndDistanceCompute for L2Expanded / L2SqrtExpanded / CosineExpanded
// metrics. Runs `fusedDistanceNNMinReduce` K times on bit-identical hardcoded
// inputs and reports whether every output (both the argmin keys and the
// distance values) is bitwise identical across the K runs.
//
// This is intentionally placed alongside the function under test so the same
// code path is exercised. Triggered once per <DataT, IndexT> instantiation
// from the first call to minClusterAndDistanceCompute via a static atomic.
template <typename DataT, typename IndexT>
void runFusedNNDeterminismSelfTest(raft::resources const& handle)
{
  constexpr IndexT n_samples  = 64;
  constexpr IndexT n_features = 4;
  constexpr IndexT n_clusters = 4;
  constexpr int K             = 4;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  auto X         = raft::make_device_matrix<DataT, IndexT>(handle, n_samples, n_features);
  auto centroids = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);

  raft::random::RngState rng_x{12345ULL};
  raft::random::RngState rng_c{67890ULL};
  raft::random::uniform(handle,
                        rng_x,
                        X.data_handle(),
                        static_cast<IndexT>(n_samples * n_features),
                        DataT{-1},
                        DataT{1});
  raft::random::uniform(handle,
                        rng_c,
                        centroids.data_handle(),
                        static_cast<IndexT>(n_clusters * n_features),
                        DataT{-1},
                        DataT{1});

  auto L2NormX       = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  auto centroidsNorm = raft::make_device_vector<DataT, IndexT>(handle, n_clusters);
  raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
    handle,
    raft::make_device_matrix_view<const DataT, IndexT>(X.data_handle(), n_samples, n_features),
    L2NormX.view());
  raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
    handle,
    raft::make_device_matrix_view<const DataT, IndexT>(
      centroids.data_handle(), n_clusters, n_features),
    centroidsNorm.view());

  rmm::device_uvector<char> workspace(sizeof(int) * static_cast<std::size_t>(n_samples), stream);

  std::vector<std::vector<raft::KeyValuePair<IndexT, DataT>>> host_outputs(K);
  for (int k = 0; k < K; ++k) {
    auto out =
      raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(handle, n_samples);
    raft::KeyValuePair<IndexT, DataT> initial_value(0, std::numeric_limits<DataT>::max());
    raft::matrix::fill(handle, out.view(), initial_value);

    cuvs::distance::fusedDistanceNNMinReduce<DataT, raft::KeyValuePair<IndexT, DataT>, IndexT>(
      out.data_handle(),
      X.data_handle(),
      centroids.data_handle(),
      L2NormX.data_handle(),
      centroidsNorm.data_handle(),
      n_samples,
      n_clusters,
      n_features,
      static_cast<void*>(workspace.data()),
      /*sqrt=*/false,
      /*initOutBuffer=*/false,
      /*isRowMajor=*/true,
      cuvs::distance::DistanceType::L2Expanded,
      0.0f,
      stream);

    host_outputs[k].resize(static_cast<std::size_t>(n_samples));
    raft::copy(host_outputs[k].data(), out.data_handle(), n_samples, stream);
    raft::resource::sync_stream(handle, stream);
  }

  // Compare every (k > 0) run against run 0 element-by-element. Both the
  // argmin key (the would-be cluster label) and the distance value are
  // checked with bitwise equality.
  int keys_diffs   = 0;
  int values_diffs = 0;
  for (int k = 1; k < K; ++k) {
    for (IndexT i = 0; i < n_samples; ++i) {
      if (host_outputs[k][i].key != host_outputs[0][i].key) { ++keys_diffs; }
      if (std::memcmp(&host_outputs[k][i].value, &host_outputs[0][i].value, sizeof(DataT)) != 0) {
        ++values_diffs;
      }
    }
  }

  RAFT_LOG_INFO(
    "fusedDistanceNNMinReduce determinism self-test (%d runs, %lld samples, %lld clusters, "
    "%lld features): keys=%s (%d diffs of %d compared), values=%s (%d bit-diffs of %d compared)",
    K,
    static_cast<long long>(n_samples),
    static_cast<long long>(n_clusters),
    static_cast<long long>(n_features),
    keys_diffs == 0 ? "ALL MATCH" : "MISMATCH",
    keys_diffs,
    static_cast<int>((K - 1) * n_samples),
    values_diffs == 0 ? "ALL MATCH" : "MISMATCH",
    values_diffs,
    static_cast<int>((K - 1) * n_samples));
}

}  // namespace

// Calculates a <key, value> pair for every sample in input 'X' where key is an
// index to an sample in 'centroids' (index of the nearest centroid) and 'value'
// is the distance between the sample and the 'centroids[key]'.
template <typename DataT, typename IndexT>
void minClusterAndDistanceCompute(
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
  bool is_fused       = metric == cuvs::distance::DistanceType::L2Expanded ||
                  metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                  metric == cuvs::distance::DistanceType::CosineExpanded;

  // ----- One-shot determinism self-test for the fused L2 NN kernel -----
  // Runs once per <DataT, IndexT> instantiation on the very first call.
  // Feeds the fused kernel bit-identical hardcoded inputs K times and checks
  // whether outputs are bitwise identical across the K runs.
  {
    static std::atomic<bool> s_self_test_done{false};
    if (!s_self_test_done.exchange(true)) {
      runFusedNNDeterminismSelfTest<DataT, IndexT>(handle);
    }
  }

  // ----- Nondeterminism diagnostic: fingerprint INPUTS (X, centroids, L2NormX) -----
  // Counterpart to the diagnostic in compute_centroid_adjustments. If calls
  // with identical INPUT hashes here produce identical OUTPUT
  // (keys_hash, values_hash), then label generation is deterministic and
  // any downstream drift is from reduce_rows_by_key atomics, not from this
  // kernel.
  static std::atomic<int> s_mcad_call_id{0};
  const int call_id = s_mcad_call_id.fetch_add(1);
  {
    const std::size_t n_X_elems =
      static_cast<std::size_t>(n_samples) * static_cast<std::size_t>(n_features);
    const std::size_t n_centroids_elems =
      static_cast<std::size_t>(n_clusters) * static_cast<std::size_t>(n_features);

    std::vector<DataT> h_X(n_X_elems);
    std::vector<DataT> h_c(n_centroids_elems);
    std::vector<DataT> h_norm(static_cast<std::size_t>(n_samples));
    raft::copy(h_X.data(), X.data_handle(), n_X_elems, stream);
    raft::copy(h_c.data(), centroids.data_handle(), n_centroids_elems, stream);
    raft::copy(h_norm.data(), L2NormX.data_handle(), n_samples, stream);
    raft::resource::sync_stream(handle, stream);

    auto fnv_fp = [](std::vector<DataT> const& buf) {
      std::uint64_t h = 1469598103934665603ULL;
      for (auto v : buf) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof(v));
        h ^= bits;
        h *= 1099511628211ULL;
      }
      return h;
    };

    RAFT_LOG_INFO(
      "minClusterAndDistanceCompute[call=%d]: INPUTS n_samples=%lld n_features=%lld "
      "n_clusters=%lld metric=%d is_fused=%d X_hash=0x%016llx centroids_hash=0x%016llx "
      "L2NormX_hash=0x%016llx",
      call_id,
      static_cast<long long>(n_samples),
      static_cast<long long>(n_features),
      static_cast<long long>(n_clusters),
      static_cast<int>(metric),
      static_cast<int>(is_fused),
      static_cast<unsigned long long>(fnv_fp(h_X)),
      static_cast<unsigned long long>(fnv_fp(h_c)),
      static_cast<unsigned long long>(fnv_fp(h_norm)));
  }
  // ----- end diagnostic for INPUTS -----

  if (is_fused) {
    L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
    auto centroidsNorm =
      raft::make_device_vector_view<DataT, IndexT>(L2NormBuf_OR_DistBuf.data(), n_clusters);

    if (metric == cuvs::distance::DistanceType::CosineExpanded) {
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        handle, centroids, centroidsNorm, raft::sqrt_op{});
    } else {
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        handle, centroids, centroidsNorm);
    }

    raft::KeyValuePair<IndexT, DataT> initial_value(0, std::numeric_limits<DataT>::max());
    raft::matrix::fill(handle, minClusterAndDistance, initial_value);

    bool should_use_fused =
      use_fused<DataT, IndexT, IndexT>(handle, n_samples, n_clusters, n_features);

    if (should_use_fused) {
      workspace.resize((sizeof(int)) * n_samples, stream);

      cuvs::distance::fusedDistanceNNMinReduce<DataT, raft::KeyValuePair<IndexT, DataT>, IndexT>(
        minClusterAndDistance.data_handle(),
        X.data_handle(),
        centroids.data_handle(),
        L2NormX.data_handle(),
        centroidsNorm.data_handle(),
        n_samples,
        n_clusters,
        n_features,
        (void*)workspace.data(),
        metric != cuvs::distance::DistanceType::L2Expanded,
        false,
        true,
        metric,
        0.0f,
        stream);
    } else {
      workspace.resize(sizeof(DataT) * n_samples * n_clusters, stream);

      cuvs::distance::
        unfusedDistanceNNMinReduce<DataT, DataT, raft::KeyValuePair<IndexT, DataT>, IndexT>(
          handle,
          minClusterAndDistance.data_handle(),
          X.data_handle(),
          centroids.data_handle(),
          L2NormX.data_handle(),
          centroidsNorm.data_handle(),
          n_samples,
          n_clusters,
          n_features,
          (void*)workspace.data(),
          metric != cuvs::distance::DistanceType::L2Expanded,
          false,
          true,
          metric,
          0.0f,
          stream);
    }
  } else {
    auto dataBatchSize      = getDataBatchSize(batch_samples, n_samples);
    auto centroidsBatchSize = getCentroidsBatchSize(batch_centroids, n_clusters);

    // TODO: Unless pool allocator is used, passing in a workspace for this
    // isn't really increasing performance because this needs to do a re-allocation
    // anyways. ref https://github.com/rapidsai/raft/issues/930
    L2NormBuf_OR_DistBuf.resize(dataBatchSize * centroidsBatchSize, stream);

    // pairwiseDistance[ns x nc] - tensor wrapper around the distance buffer
    auto pairwiseDistance = raft::make_device_matrix_view<DataT, IndexT>(
      L2NormBuf_OR_DistBuf.data(), dataBatchSize, centroidsBatchSize);

    raft::KeyValuePair<IndexT, DataT> initial_value(0, std::numeric_limits<DataT>::max());
    raft::matrix::fill(handle, minClusterAndDistance, initial_value);

    // tile over the input dataset
    for (IndexT dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
      // # of samples for the current batch
      auto ns = std::min((IndexT)dataBatchSize, n_samples - dIdx);

      // datasetView [ns x n_features] - view representing the current batch of
      // input dataset
      auto datasetView = raft::make_device_matrix_view<const DataT, IndexT>(
        X.data_handle() + (dIdx * n_features), ns, n_features);

      // minClusterAndDistanceView [ns x n_clusters]
      auto minClusterAndDistanceView =
        raft::make_device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT>(
          minClusterAndDistance.data_handle() + dIdx, ns);

      // tile over the centroids
      for (IndexT cIdx = 0; cIdx < n_clusters; cIdx += centroidsBatchSize) {
        // # of centroids for the current batch
        auto nc = std::min((IndexT)centroidsBatchSize, n_clusters - cIdx);

        // centroidsView [nc x n_features] - view representing the current batch
        // of centroids
        auto centroidsView = raft::make_device_matrix_view<const DataT, IndexT>(
          centroids.data_handle() + (cIdx * n_features), nc, n_features);

        // pairwiseDistanceView [ns x nc] - view representing the pairwise
        // distance for current batch
        auto pairwiseDistanceView =
          raft::make_device_matrix_view<DataT, IndexT>(pairwiseDistance.data_handle(), ns, nc);

        // calculate pairwise distance between current tile of cluster centroids
        // and input dataset
        pairwise_distance_kmeans<DataT, IndexT>(
          handle, datasetView, centroidsView, pairwiseDistanceView, metric);

        // argmin reduction returning <index, value> pair
        // calculates the closest centroid and the distance to the closest
        // centroid
        raft::linalg::coalescedReduction(
          minClusterAndDistanceView.data_handle(),
          pairwiseDistanceView.data_handle(),
          pairwiseDistanceView.extent(1),
          pairwiseDistanceView.extent(0),
          initial_value,
          stream,
          true,
          [=] __device__(const DataT val, const IndexT i) {
            raft::KeyValuePair<IndexT, DataT> pair;
            pair.key   = cIdx + i;
            pair.value = val;
            return pair;
          },
          raft::argmin_op{},
          raft::identity_op{});
      }
    }
  }

  // ----- Nondeterminism diagnostic: fingerprint OUTPUTS (keys, values) -----
  // Split the KeyValuePair vector into keys (cluster indices = future labels)
  // and values (distances). The keys are the inputs to
  // compute_centroid_adjustments downstream.
  {
    std::vector<raft::KeyValuePair<IndexT, DataT>> h_kvp(static_cast<std::size_t>(n_samples));
    raft::copy(h_kvp.data(), minClusterAndDistance.data_handle(), n_samples, stream);
    raft::resource::sync_stream(handle, stream);

    std::uint64_t keys_hash   = 1469598103934665603ULL;
    std::uint64_t values_hash = 1469598103934665603ULL;
    for (auto const& kv : h_kvp) {
      keys_hash ^= static_cast<std::uint64_t>(kv.key);
      keys_hash *= 1099511628211ULL;
      std::uint64_t vbits = 0;
      std::memcpy(&vbits, &kv.value, sizeof(kv.value));
      values_hash ^= vbits;
      values_hash *= 1099511628211ULL;
    }

    std::string keys_head;
    std::string values_head;
    const std::size_t hd = std::min<std::size_t>(8, h_kvp.size());
    for (std::size_t i = 0; i < hd; ++i) {
      if (i) {
        keys_head += ',';
        values_head += ',';
      }
      keys_head += std::to_string(static_cast<long long>(h_kvp[i].key));
      values_head += std::to_string(static_cast<double>(h_kvp[i].value));
    }

    RAFT_LOG_INFO(
      "minClusterAndDistanceCompute[call=%d]: OUTPUTS keys_hash=0x%016llx values_hash=0x%016llx "
      "keys_head=[%s] values_head=[%s]",
      call_id,
      static_cast<unsigned long long>(keys_hash),
      static_cast<unsigned long long>(values_hash),
      keys_head.c_str(),
      values_head.c_str());
  }
  // ----- end diagnostic for OUTPUTS -----
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
void minClusterDistanceCompute(raft::resources const& handle,
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
                  metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                  metric == cuvs::distance::DistanceType::CosineExpanded;

  raft::matrix::fill(handle, minClusterDistance, std::numeric_limits<DataT>::max());

  if (is_fused) {
    L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
    auto centroidsNorm =
      raft::make_device_vector_view<DataT, IndexT>(L2NormBuf_OR_DistBuf.data(), n_clusters);

    if (metric == cuvs::distance::DistanceType::CosineExpanded) {
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<const DataT, IndexT>(
          centroids.data_handle(), centroids.extent(0), centroids.extent(1)),
        centroidsNorm,
        raft::sqrt_op{});
    } else {
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        handle,
        raft::make_device_matrix_view<const DataT, IndexT>(
          centroids.data_handle(), centroids.extent(0), centroids.extent(1)),
        centroidsNorm);
    }

    workspace.resize(sizeof(int) * n_samples, stream);

    cuvs::distance::fusedDistanceNNMinReduce<DataT, DataT, IndexT>(
      minClusterDistance.data_handle(),
      X.data_handle(),
      centroids.data_handle(),
      L2NormX.data_handle(),
      centroidsNorm.data_handle(),
      n_samples,
      n_clusters,
      n_features,
      (void*)workspace.data(),
      metric != cuvs::distance::DistanceType::L2Expanded,
      false,
      true,
      metric,
      0.0f,
      stream);
  } else {
    auto dataBatchSize      = getDataBatchSize(batch_samples, n_samples);
    auto centroidsBatchSize = getCentroidsBatchSize(batch_centroids, n_clusters);

    L2NormBuf_OR_DistBuf.resize(dataBatchSize * centroidsBatchSize, stream);

    auto pairwiseDistance = raft::make_device_matrix_view<DataT, IndexT>(
      L2NormBuf_OR_DistBuf.data(), dataBatchSize, centroidsBatchSize);

    // tile over the input data and calculate distance matrix [n_samples x
    // n_clusters]
    for (IndexT dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
      auto ns = std::min((IndexT)dataBatchSize, n_samples - dIdx);

      auto datasetView = raft::make_device_matrix_view<const DataT, IndexT>(
        X.data_handle() + dIdx * n_features, ns, n_features);

      auto minClusterDistanceView =
        raft::make_device_vector_view<DataT, IndexT>(minClusterDistance.data_handle() + dIdx, ns);

      // tile over the centroids
      for (IndexT cIdx = 0; cIdx < n_clusters; cIdx += centroidsBatchSize) {
        auto nc = std::min((IndexT)centroidsBatchSize, n_clusters - cIdx);

        auto centroidsView = raft::make_device_matrix_view<DataT, IndexT>(
          centroids.data_handle() + cIdx * n_features, nc, n_features);

        auto pairwiseDistanceView =
          raft::make_device_matrix_view<DataT, IndexT>(pairwiseDistance.data_handle(), ns, nc);

        pairwise_distance_kmeans<DataT, IndexT>(
          handle, datasetView, centroidsView, pairwiseDistanceView, metric);

        raft::linalg::coalescedReduction(minClusterDistanceView.data_handle(),
                                         pairwiseDistanceView.data_handle(),
                                         pairwiseDistanceView.extent(1),
                                         pairwiseDistanceView.extent(0),
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
