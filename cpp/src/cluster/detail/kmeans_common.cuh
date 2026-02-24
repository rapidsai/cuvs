/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../distance/distance.cuh"
#include "../../distance/fused_distance_nn.cuh"
#include <cstdint>
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/random/permute.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_histogram.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/arg_index_input_iterator.cuh>
#include <cuda.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <optional>
#include <random>

namespace cuvs::cluster::kmeans::detail {

template <typename DataT, typename IndexT>
struct SamplingOp {
  DataT* rnd;
  uint8_t* flag;
  DataT cluster_cost;
  double oversampling_factor;
  IndexT n_clusters;

  CUB_RUNTIME_FUNCTION __forceinline__
  SamplingOp(DataT c, double l, IndexT k, DataT* rand, uint8_t* ptr)
    : cluster_cost(c), oversampling_factor(l), n_clusters(k), rnd(rand), flag(ptr)
  {
  }

  __host__ __device__ __forceinline__ bool operator()(
    const raft::KeyValuePair<ptrdiff_t, DataT>& a) const
  {
    DataT prob_threshold = (DataT)rnd[a.key];

    DataT prob_x = ((oversampling_factor * n_clusters * a.value) / cluster_cost);

    return !flag[a.key] && (prob_x > prob_threshold);
  }
};

template <typename IndexT, typename DataT>
struct KeyValueIndexOp {
  __host__ __device__ __forceinline__ IndexT
  operator()(const raft::KeyValuePair<IndexT, DataT>& a) const
  {
    return a.key;
  }
};

// Computes the intensity histogram from a sequence of labels
template <typename SampleIteratorT, typename CounterT, typename IndexT>
void countLabels(raft::resources const& handle,
                 SampleIteratorT labels,
                 CounterT* count,
                 IndexT n_samples,
                 IndexT n_clusters,
                 rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  // CUB::DeviceHistogram requires a signed index type
  typedef typename std::make_signed_t<IndexT> CubIndexT;

  CubIndexT num_levels  = n_clusters + 1;
  CubIndexT lower_level = 0;
  CubIndexT upper_level = n_clusters;

  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(nullptr,
                                                    temp_storage_bytes,
                                                    labels,
                                                    count,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    static_cast<CubIndexT>(n_samples),
                                                    stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(workspace.data(),
                                                    temp_storage_bytes,
                                                    labels,
                                                    count,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    static_cast<CubIndexT>(n_samples),
                                                    stream));
}

template <typename DataT, typename IndexT>
void checkWeight(raft::resources const& handle,
                 raft::device_vector_view<DataT, IndexT> weight,
                 rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto wt_aggr        = raft::make_device_scalar<DataT>(handle, 0);
  auto n_samples      = weight.extent(0);

  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceReduce::Sum(
    nullptr, temp_storage_bytes, weight.data_handle(), wt_aggr.data_handle(), n_samples, stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceReduce::Sum(workspace.data(),
                                       temp_storage_bytes,
                                       weight.data_handle(),
                                       wt_aggr.data_handle(),
                                       n_samples,
                                       stream));
  DataT wt_sum = 0;
  raft::copy(handle,
             raft::make_host_scalar_view(&wt_sum),
             raft::make_device_scalar_view(wt_aggr.data_handle()));
  raft::resource::sync_stream(handle, stream);

  if (wt_sum != n_samples) {
    RAFT_LOG_DEBUG(
      "[Warning!] KMeans: normalizing the user provided sample weight to "
      "sum up to %d samples",
      n_samples);

    auto scale = static_cast<DataT>(n_samples) / wt_sum;
    raft::linalg::map(
      handle, weight, raft::mul_const_op<DataT>{scale}, raft::make_const_mdspan(weight));
  }
}

template <typename IndexT>
IndexT getDataBatchSize(int batch_samples, IndexT n_samples)
{
  auto minVal = std::min(static_cast<IndexT>(batch_samples), n_samples);
  return (minVal == 0) ? n_samples : minVal;
}

template <typename IndexT>
IndexT getCentroidsBatchSize(int batch_centroids, IndexT n_local_clusters)
{
  auto minVal = std::min(static_cast<IndexT>(batch_centroids), n_local_clusters);
  return (minVal == 0) ? n_local_clusters : minVal;
}

template <typename InputT,
          typename OutputT,
          typename MainOpT,
          typename ReductionOpT,
          typename IndexT = int>
void computeClusterCost(raft::resources const& handle,
                        raft::device_vector_view<InputT, IndexT> minClusterDistance,
                        rmm::device_uvector<char>& workspace,
                        raft::device_scalar_view<OutputT> clusterCost,
                        MainOpT main_op,
                        ReductionOpT reduction_op)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  thrust::transform_iterator<MainOpT, InputT*> itr(minClusterDistance.data_handle(), main_op);

  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(nullptr,
                                          temp_storage_bytes,
                                          itr,
                                          clusterCost.data_handle(),
                                          minClusterDistance.size(),
                                          reduction_op,
                                          OutputT(),
                                          stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(workspace.data(),
                                          temp_storage_bytes,
                                          itr,
                                          clusterCost.data_handle(),
                                          minClusterDistance.size(),
                                          reduction_op,
                                          OutputT(),
                                          stream));
}

template <typename DataT, typename IndexT>
void sampleCentroids(raft::resources const& handle,
                     raft::device_matrix_view<const DataT, IndexT> X,
                     raft::device_vector_view<DataT, IndexT> minClusterDistance,
                     raft::device_vector_view<uint8_t, IndexT> isSampleCentroid,
                     SamplingOp<DataT, IndexT>& select_op,
                     rmm::device_uvector<DataT>& inRankCp,
                     rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream  = raft::resource::get_cuda_stream(handle);
  auto n_local_samples = X.extent(0);
  auto n_features      = X.extent(1);

  auto nSelected = raft::make_device_scalar<IndexT>(handle, 0);
  cub::ArgIndexInputIterator<DataT*> ip_itr(minClusterDistance.data_handle());
  auto sampledMinClusterDistance =
    raft::make_device_vector<raft::KeyValuePair<ptrdiff_t, DataT>, IndexT>(handle, n_local_samples);
  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceSelect::If(nullptr,
                                      temp_storage_bytes,
                                      ip_itr,
                                      sampledMinClusterDistance.data_handle(),
                                      nSelected.data_handle(),
                                      n_local_samples,
                                      select_op,
                                      stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceSelect::If(workspace.data(),
                                      temp_storage_bytes,
                                      ip_itr,
                                      sampledMinClusterDistance.data_handle(),
                                      nSelected.data_handle(),
                                      n_local_samples,
                                      select_op,
                                      stream));

  IndexT nPtsSampledInRank = 0;
  raft::copy(handle,
             raft::make_host_scalar_view(&nPtsSampledInRank),
             raft::make_device_scalar_view(nSelected.data_handle()));
  raft::resource::sync_stream(handle, stream);

  uint8_t* rawPtr_isSampleCentroid = isSampleCentroid.data_handle();
  thrust::for_each_n(raft::resource::get_thrust_policy(handle),
                     sampledMinClusterDistance.data_handle(),
                     nPtsSampledInRank,
                     [=] __device__(raft::KeyValuePair<ptrdiff_t, DataT> val) {
                       rawPtr_isSampleCentroid[val.key] = 1;
                     });

  inRankCp.resize(nPtsSampledInRank * n_features, stream);

  raft::matrix::gather((DataT*)X.data_handle(),
                       X.extent(1),
                       X.extent(0),
                       sampledMinClusterDistance.data_handle(),
                       nPtsSampledInRank,
                       inRankCp.data(),
                       raft::key_op{},
                       stream);
}

// calculate pairwise distance between 'dataset[n x d]' and 'centroids[k x d]',
// result will be stored in 'pairwiseDistance[n x k]'
template <typename DataT, typename IndexT>
void pairwise_distance_kmeans(raft::resources const& handle,
                              raft::device_matrix_view<const DataT, IndexT> X,
                              raft::device_matrix_view<const DataT, IndexT> centroids,
                              raft::device_matrix_view<DataT, IndexT> pairwiseDistance,
                              cuvs::distance::DistanceType metric)
{
  auto n_samples  = X.extent(0);
  auto n_features = X.extent(1);
  auto n_clusters = centroids.extent(0);

  ASSERT(X.extent(1) == centroids.extent(1),
         "# features in dataset and centroids are different (must be same)");

  if (metric == cuvs::distance::DistanceType::L2Expanded) {
    cuvs::distance::distance<cuvs::distance::DistanceType::L2Expanded,
                             DataT,
                             DataT,
                             DataT,
                             raft::layout_c_contiguous,
                             IndexT>(handle, X, centroids, pairwiseDistance);
  } else if (metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    cuvs::distance::distance<cuvs::distance::DistanceType::L2SqrtExpanded,
                             DataT,
                             DataT,
                             DataT,
                             raft::layout_c_contiguous,
                             IndexT>(handle, X, centroids, pairwiseDistance);
  } else {
    RAFT_FAIL("kmeans requires L2Expanded or L2SqrtExpanded distance, have %i",
              static_cast<int>(metric));
  }
}

// shuffle and randomly select 'n_samples_to_gather' from input 'in' and stores
// in 'out' does not modify the input
template <typename DataT, typename IndexT>
void shuffleAndGather(raft::resources const& handle,
                      raft::device_matrix_view<const DataT, IndexT> in,
                      raft::device_matrix_view<DataT, IndexT> out,
                      uint32_t n_samples_to_gather,
                      uint64_t seed)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = in.extent(0);
  auto n_features     = in.extent(1);

  auto indices = raft::make_device_vector<IndexT, IndexT>(handle, n_samples);

  // shuffle indices on device
  raft::random::permute<DataT, IndexT, IndexT>(indices.data_handle(),
                                               nullptr,
                                               nullptr,
                                               (IndexT)in.extent(1),
                                               (IndexT)in.extent(0),
                                               true,
                                               stream);

  raft::matrix::gather((DataT*)in.data_handle(),
                       in.extent(1),
                       in.extent(0),
                       indices.data_handle(),
                       static_cast<IndexT>(n_samples_to_gather),
                       out.data_handle(),
                       stream);
}

// Calculates a <key, value> pair for every sample in input 'X' where key is an
// index to an sample in 'centroids' (index of the nearest centroid) and 'value'
// is the distance between the sample and the 'centroid[key]'
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
  rmm::device_uvector<char>& workspace);

#define EXTERN_TEMPLATE_MIN_CLUSTER_AND_DISTANCE(DataT, IndexT)                                \
  extern template void minClusterAndDistanceCompute<DataT, IndexT>(                            \
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

EXTERN_TEMPLATE_MIN_CLUSTER_AND_DISTANCE(float, int64_t)
EXTERN_TEMPLATE_MIN_CLUSTER_AND_DISTANCE(float, int)
EXTERN_TEMPLATE_MIN_CLUSTER_AND_DISTANCE(double, int64_t)
EXTERN_TEMPLATE_MIN_CLUSTER_AND_DISTANCE(double, int)

#undef EXTERN_TEMPLATE_MIN_CLUSTER_AND_DISTANCE

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
                               rmm::device_uvector<char>& workspace);

#define EXTERN_TEMPLATE_MIN_CLUSTER_DISTANCE(DataT, IndexT)      \
  extern template void minClusterDistanceCompute<DataT, IndexT>( \
    raft::resources const& handle,                               \
    raft::device_matrix_view<const DataT, IndexT> X,             \
    raft::device_matrix_view<DataT, IndexT> centroids,           \
    raft::device_vector_view<DataT, IndexT> minClusterDistance,  \
    raft::device_vector_view<DataT, IndexT> L2NormX,             \
    rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,            \
    cuvs::distance::DistanceType metric,                         \
    int batch_samples,                                           \
    int batch_centroids,                                         \
    rmm::device_uvector<char>& workspace);

EXTERN_TEMPLATE_MIN_CLUSTER_DISTANCE(float, int64_t)
EXTERN_TEMPLATE_MIN_CLUSTER_DISTANCE(double, int64_t)
EXTERN_TEMPLATE_MIN_CLUSTER_DISTANCE(float, int)
EXTERN_TEMPLATE_MIN_CLUSTER_DISTANCE(double, int)

#undef EXTERN_TEMPLATE_MIN_CLUSTER_DISTANCE

template <typename DataT, typename IndexT>
void countSamplesInCluster(raft::resources const& handle,
                           const cuvs::cluster::kmeans::params& params,
                           raft::device_matrix_view<const DataT, IndexT> X,
                           raft::device_vector_view<const DataT, IndexT> L2NormX,
                           raft::device_matrix_view<DataT, IndexT> centroids,
                           rmm::device_uvector<char>& workspace,
                           raft::device_vector_view<DataT, IndexT> sampleCountInCluster)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = centroids.extent(0);

  // stores (key, value) pair corresponding to each sample where
  //   - key is the index of nearest cluster
  //   - value is the distance to the nearest cluster
  auto minClusterAndDistance =
    raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(handle, n_samples);

  // temporary buffer to store distance matrix, destructor releases the resource
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // computes minClusterAndDistance[0:n_samples) where  minClusterAndDistance[i]
  // is a <key, value> pair where
  //   'key' is index to an sample in 'centroids' (index of the nearest
  //   centroid) and 'value' is the distance between the sample 'X[i]' and the
  //   'centroid[key]'
  cuvs::cluster::kmeans::detail::minClusterAndDistanceCompute(
    handle,
    X,
    (raft::device_matrix_view<const DataT, IndexT>)centroids,
    minClusterAndDistance.view(),
    L2NormX,
    L2NormBuf_OR_DistBuf,
    params.metric,
    params.batch_samples,
    params.batch_centroids,
    workspace);

  // Using TransformInputIteratorT to dereference an array of raft::KeyValuePair
  // and converting them to just return the Key to be used in reduce_rows_by_key
  // prims
  cuvs::cluster::kmeans::detail::KeyValueIndexOp<IndexT, DataT> conversion_op;
  thrust::transform_iterator<cuvs::cluster::kmeans::detail::KeyValueIndexOp<IndexT, DataT>,
                             raft::KeyValuePair<IndexT, DataT>*>
    itr(minClusterAndDistance.data_handle(), conversion_op);

  // count # of samples in each cluster
  countLabels(handle,
              itr,
              sampleCountInCluster.data_handle(),
              (IndexT)n_samples,
              (IndexT)n_clusters,
              workspace);
}
}  // namespace cuvs::cluster::kmeans::detail
