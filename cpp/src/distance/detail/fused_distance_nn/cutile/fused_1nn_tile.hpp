/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <type_traits>

#include <cuda_runtime.h>
#include <raft/core/kvp.hpp>

#include <cuvs/distance/distance.hpp>

namespace cuvs {
namespace distance {
namespace detail {

template <typename OutT, typename IdxT, typename DataT>
inline constexpr bool is_fused_1nn_kvp_output_v =
  (std::is_same_v<DataT, float> || std::is_same_v<DataT, half>) &&
  (std::is_same_v<OutT, raft::KeyValuePair<IdxT, float>> ||
   std::is_same_v<OutT, raft::KeyValuePair<IdxT, DataT>>);

template <typename OutT, typename IdxT, typename DataT>
concept Fused1nnKvpOutput = is_fused_1nn_kvp_output_v<OutT, IdxT, DataT>;

template <typename DataT, typename OutT, typename IdxT>
  requires Fused1nnKvpOutput<OutT, IdxT, DataT>
bool try_fused_1nn_tile(OutT* min,
                        const DataT* x,
                        const DataT* y,
                        const DataT* xn,
                        const DataT* yn,
                        IdxT m,
                        IdxT n,
                        IdxT k,
                        cuvs::distance::DistanceType metric,
                        bool is_sqrt,
                        cudaStream_t stream);

template <typename DataT, typename OutT, typename IdxT>
  requires(!Fused1nnKvpOutput<OutT, IdxT, DataT>)
bool try_fused_1nn_tile(OutT*,
                        const DataT*,
                        const DataT*,
                        const DataT*,
                        const DataT*,
                        IdxT,
                        IdxT,
                        IdxT,
                        cuvs::distance::DistanceType,
                        bool,
                        cudaStream_t)
{
  return false;
}

}  // namespace detail
}  // namespace distance
}  // namespace cuvs
