/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_scalar.hpp>

#include <cub/cub.cuh>

#include <cmath>

#include <algorithm>
#include <iostream>
#include <numeric>

namespace cuvs {
namespace stats {
namespace detail {

/**
 * @brief kernel that calculates the average intra-cluster distance for every sample data point and
 * updates the cluster distance to max value
 * @tparam DataT: type of the data samples
 * @tparam label_t: type of the labels
 * @param sample_to_cluster_sum_of_distances: the pointer to the 2D array that contains the sum of
 * distances from every sample to every cluster (nRows x nLabels)
 * @param bin_count_array: pointer to the 1D array that contains the count of samples per cluster (1
 * x nLabels)
 * @param d_a_array: the pointer to the array of average intra-cluster distances for every sample in
 * device memory (1 x nRows)
 * @param labels: the pointer to the array containing labels for every data sample (1 x nRows)
 * @param nRows: number of data samples
 * @param nLabels: number of Labels
 * @param MAX_VAL: DataT specific upper limit
 */
template <typename DataT, typename label_t>
RAFT_KERNEL populate_a_kernel(DataT* sample_to_cluster_sum_of_distances,
                              DataT* bin_count_array,
                              DataT* d_a_array,
                              const label_t* labels,
                              int nRows,
                              int nLabels,
                              const DataT MAX_VAL)
{
  // getting the current index
  int sample_index = threadIdx.x + blockIdx.x * blockDim.x;

  if (sample_index >= nRows) return;

  // sampleDistanceVector is an array that stores that particular row of the distance_matrix
  DataT* sample_to_cluster_sum_of_distances_vector =
    &sample_to_cluster_sum_of_distances[sample_index * nLabels];

  label_t sample_cluster = labels[sample_index];

  int sample_cluster_index = static_cast<int>(sample_cluster);

  if (bin_count_array[sample_cluster_index] - 1 <= 0) {
    d_a_array[sample_index] = -1;
    return;

  }

  else {
    d_a_array[sample_index] = (sample_to_cluster_sum_of_distances_vector[sample_cluster_index]) /
                              (bin_count_array[sample_cluster_index] - 1);

    // modifying the sampleDistanceVector to give sample average distance
    sample_to_cluster_sum_of_distances_vector[sample_cluster_index] = MAX_VAL;
  }
}

/**
 * @brief function to calculate the bincounts of number of samples in every label
 * @tparam DataT: type of the data samples
 * @tparam label_t: type of the labels
 * @param labels: the pointer to the array containing labels for every data sample (1 x nRows)
 * @param bin_count_array: pointer to the 1D array that contains the count of samples per cluster (1
 * x nLabels)
 * @param nRows: number of data samples
 * @param nUniqueLabels: number of Labels
 * @param workspace: device buffer containing workspace memory
 * @param stream: the cuda stream where to launch this kernel
 */
template <typename DataT, typename label_t>
void count_labels(const label_t* labels,
                  DataT* bin_count_array,
                  int nRows,
                  int nUniqueLabels,
                  rmm::device_uvector<char>& workspace,
                  cudaStream_t stream)
{
  int num_levels            = nUniqueLabels + 1;
  label_t lower_level       = 0;
  label_t upper_level       = nUniqueLabels;
  size_t temp_storage_bytes = 0;

  rmm::device_uvector<int> count_array(nUniqueLabels, stream);

  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(nullptr,
                                                    temp_storage_bytes,
                                                    labels,
                                                    bin_count_array,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    nRows,
                                                    stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(workspace.data(),
                                                    temp_storage_bytes,
                                                    labels,
                                                    bin_count_array,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    nRows,
                                                    stream));
}

/**
 * @brief structure that defines the division Lambda for elementwise op
 */
template <typename DataT>
struct div_op {
  HDI auto operator()(DataT a, int b, int c) -> DataT
  {
    if (b == 0) {
      return ULLONG_MAX;
    } else {
      return a / b;
    }
  }
};

/**
 * @brief structure that defines the elementwise operation to calculate silhouette score using
 * params 'a' and 'b'
 */
template <typename DataT>
struct sil_op {
  HDI auto operator()(DataT a, DataT b) -> DataT
  {
    if (a == 0 && b == 0 || a == b) {
      return 0;
    } else if (a == -1) {
      return 0;
    } else if (a > b) {
      return (b - a) / a;
    } else {
      return (b - a) / b;
    }
  }
};

/**
 * @brief main function that returns the average silhouette score for a given set of data and its
 * clusterings
 * @tparam DataT: type of the data samples
 * @tparam label_t: type of the labels
 * @param X_in: pointer to the input Data samples array (nRows x nCols)
 * @param nRows: number of data samples
 * @param nCols: number of features
 * @param labels: the pointer to the array containing labels for every data sample (1 x nRows)
 * @param nLabels: number of Labels
 * @param silhouette_scorePerSample: pointer to the array that is optionally taken in as input and
 * is populated with the silhouette score for every sample (1 x nRows)
 * @param stream: the cuda stream where to launch this kernel
 * @param metric: the numerical value that maps to the type of distance metric to be used in the
 * calculations
 */
template <typename DataT, typename label_t>
auto silhouette_score(
  raft::resources const& handle,
  const DataT* X_in,
  int nRows,
  int nCols,
  const label_t* labels,
  int nLabels,
  DataT* silhouette_scorePerSample,
  cudaStream_t stream,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded) -> DataT
{
  ASSERT(nLabels >= 2 && nLabels <= (nRows - 1),
         "silhouette Score not defined for the given number of labels!");

  // compute the distance matrix
  rmm::device_uvector<DataT> distance_matrix(nRows * nRows, stream);
  rmm::device_uvector<char> workspace(1, stream);

  auto x_in_view = raft::make_device_matrix_view<const DataT, int64_t>(X_in, nRows, nCols);

  cuvs::distance::pairwise_distance(
    handle,
    x_in_view,
    x_in_view,
    raft::make_device_matrix_view<DataT, int64_t>(distance_matrix.data(), nRows, nRows),
    metric);

  // deciding on the array of silhouette scores for each dataPoint
  rmm::device_uvector<DataT> silhouette_score_samples(0, stream);
  DataT* per_sample_sil_score = nullptr;
  if (silhouette_scorePerSample == nullptr) {
    silhouette_score_samples.resize(nRows, stream);
    per_sample_sil_score = silhouette_score_samples.data();
  } else {
    per_sample_sil_score = silhouette_scorePerSample;
  }
  RAFT_CUDA_TRY(cudaMemsetAsync(per_sample_sil_score, 0, nRows * sizeof(DataT), stream));

  // getting the sample count per cluster
  rmm::device_uvector<DataT> bin_count_array(nLabels, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(bin_count_array.data(), 0, nLabels * sizeof(DataT), stream));
  count_labels(labels, bin_count_array.data(), nRows, nLabels, workspace, stream);

  // calculating the sample-cluster-distance-sum-array
  rmm::device_uvector<DataT> sample_to_cluster_sum_of_distances(nRows * nLabels, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(
    sample_to_cluster_sum_of_distances.data(), 0, nRows * nLabels * sizeof(DataT), stream));
  raft::linalg::reduce_cols_by_key(distance_matrix.data(),
                                   labels,
                                   sample_to_cluster_sum_of_distances.data(),
                                   nRows,
                                   nRows,
                                   nLabels,
                                   stream);

  // creating the a array and b array
  rmm::device_uvector<DataT> d_a_array(nRows, stream);
  rmm::device_uvector<DataT> d_b_array(nRows, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(d_a_array.data(), 0, nRows * sizeof(DataT), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(d_b_array.data(), 0, nRows * sizeof(DataT), stream));

  // kernel that populates the d_a_array
  // kernel configuration
  dim3 num_threads_per_block(32, 1, 1);
  dim3 num_blocks(raft::ceildiv<int>(nRows, num_threads_per_block.x), 1, 1);

  // calling the kernel
  populate_a_kernel<<<num_blocks, num_threads_per_block, 0, stream>>>(
    sample_to_cluster_sum_of_distances.data(),
    bin_count_array.data(),
    d_a_array.data(),
    labels,
    nRows,
    nLabels,
    std::numeric_limits<DataT>::max());

  // elementwise dividing by bincounts
  rmm::device_uvector<DataT> average_distance_between_sample_and_cluster(nRows * nLabels, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(average_distance_between_sample_and_cluster.data(),
                                0,
                                nRows * nLabels * sizeof(DataT),
                                stream));

  auto average_distance_between_sample_and_cluster_view = raft::make_device_matrix_view<DataT>(
    average_distance_between_sample_and_cluster.data(), nRows, nLabels);
  auto sample_to_cluster_sum_of_distances_view = raft::make_device_matrix_view<const DataT>(
    sample_to_cluster_sum_of_distances.data(), nRows, nLabels);
  auto bin_count_array_view =
    raft::make_device_vector_view<const DataT>(bin_count_array.data(), nLabels);

  raft::linalg::matrix_vector_op<raft::Apply::ALONG_ROWS>(
    handle,
    sample_to_cluster_sum_of_distances_view,
    bin_count_array_view,
    average_distance_between_sample_and_cluster_view,
    [] __device__(DataT a, DataT b) -> DataT {
      if (b == 0) {
        return static_cast<DataT>(ULLONG_MAX);
      } else {
        return a / b;
      }
    });

  // calculating row-wise minimum
  raft::linalg::reduce<true, true, DataT, DataT, int, raft::identity_op, raft::min_op>(
    d_b_array.data(),
    average_distance_between_sample_and_cluster.data(),
    nLabels,
    nRows,
    std::numeric_limits<DataT>::max(),
    stream,
    false,
    raft::identity_op{},
    raft::min_op{});

  // calculating the silhouette score per sample using the d_a_array and d_b_array
  raft::linalg::binaryOp<DataT, sil_op<DataT>>(
    per_sample_sil_score, d_a_array.data(), d_b_array.data(), nRows, sil_op<DataT>(), stream);

  // calculating the sum of all the silhouette score
  rmm::device_scalar<DataT> d_avg_silhouette_score(stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(d_avg_silhouette_score.data(), 0, sizeof(DataT), stream));

  raft::linalg::mapThenSumReduce<DataT, raft::identity_op>(d_avg_silhouette_score.data(),
                                                           nRows,
                                                           raft::identity_op(),
                                                           stream,
                                                           per_sample_sil_score,
                                                           per_sample_sil_score);

  DataT avg_silhouette_score = d_avg_silhouette_score.value(stream);

  raft::resource::sync_stream(handle, stream);

  avg_silhouette_score /= nRows;

  return avg_silhouette_score;
}

};  // namespace detail
};  // namespace stats
};  // namespace cuvs
