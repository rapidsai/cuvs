/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cub_histogram.hpp"

#include <cub/device/device_histogram.cuh>
#include <raft/util/cudart_utils.hpp>

namespace cuvs::detail {

template <typename CounterT, typename IndexT>
void count_labels(const uint32_t* labels,
                  CounterT* counts,
                  IndexT n_samples,
                  IndexT n_clusters,
                  rmm::device_uvector<char>& workspace,
                  cudaStream_t stream)
{
  IndexT num_levels         = n_clusters + 1;
  IndexT lower_level        = 0;
  IndexT upper_level        = n_clusters;
  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(nullptr,
                                                    temp_storage_bytes,
                                                    labels,
                                                    counts,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    n_samples,
                                                    stream));
  workspace.resize(temp_storage_bytes, stream);
  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(workspace.data(),
                                                    temp_storage_bytes,
                                                    labels,
                                                    counts,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    n_samples,
                                                    stream));
}

template void count_labels<uint32_t, int>(
  const uint32_t*, uint32_t*, int, int, rmm::device_uvector<char>&, cudaStream_t);
template void count_labels<uint32_t, int64_t>(
  const uint32_t*, uint32_t*, int64_t, int64_t, rmm::device_uvector<char>&, cudaStream_t);
template void count_labels<unsigned long long, int>(
  const uint32_t*, unsigned long long*, int, int, rmm::device_uvector<char>&, cudaStream_t);
template void count_labels<unsigned long long, int64_t>(
  const uint32_t*, unsigned long long*, int64_t, int64_t, rmm::device_uvector<char>&, cudaStream_t);

}  // namespace cuvs::detail

namespace cuvs::detail {
template <typename SampleIt, typename CounterT, typename LevelT, typename NumItems>
cudaError_t histogram_even(void* storage,
                           size_t& bytes,
                           SampleIt samples,
                           CounterT* counts,
                           int levels,
                           LevelT lower,
                           LevelT upper,
                           NumItems count,
                           cudaStream_t stream)
{
  return cub::DeviceHistogram::HistogramEven(
    storage, bytes, samples, counts, levels, lower, upper, count, stream);
}

// EXPLICIT_HOST_INSTANTIATIONS

}  // namespace cuvs::detail
