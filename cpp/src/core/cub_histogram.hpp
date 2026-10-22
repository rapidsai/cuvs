/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <rmm/device_uvector.hpp>

namespace cuvs::detail {

/** CUB label histograms shared by balanced k-means instantiations. */
template <typename CounterT, typename IndexT>
void count_labels(const uint32_t* labels,
                  CounterT* counts,
                  IndexT n_samples,
                  IndexT n_clusters,
                  rmm::device_uvector<char>& workspace,
                  cudaStream_t stream);

}  // namespace cuvs::detail
