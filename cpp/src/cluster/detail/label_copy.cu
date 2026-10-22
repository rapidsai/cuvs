/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans_common.cuh"
#include <cuvs/core/cuda_fp16.hpp>

namespace cuvs::cluster::kmeans::detail {
template void copyClusterLabels<int, float, int>(const raft::resources&,
                                                 const MinClusterAndDistanceResult<float, int>&,
                                                 int*);
template void copyClusterLabels<int, double, int>(const raft::resources&,
                                                  const MinClusterAndDistanceResult<double, int>&,
                                                  int*);
template void copyClusterLabels<int, float, int64_t>(
  const raft::resources&, const MinClusterAndDistanceResult<float, int64_t>&, int*);
template void copyClusterLabels<int64_t, float, int64_t>(
  const raft::resources&, const MinClusterAndDistanceResult<float, int64_t>&, int64_t*);
template void copyClusterLabels<int64_t, double, int64_t>(
  const raft::resources&, const MinClusterAndDistanceResult<double, int64_t>&, int64_t*);
template void copyClusterLabels<uint32_t, float, int64_t>(
  const raft::resources&, const MinClusterAndDistanceResult<float, int64_t>&, uint32_t*);
}  // namespace cuvs::cluster::kmeans::detail
