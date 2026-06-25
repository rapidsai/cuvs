/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <type_traits>

#include <cuda_runtime.h>
#include <raft/core/kvp.hpp>

#include <cuvs/detail/jit_lto/tileir_compat.hpp>
#include <cuvs/distance/distance.hpp>

#ifndef CUVS_CUTILE_ENABLED
#define CUVS_CUTILE_ENABLED 0
#endif

namespace cuvs {
namespace distance {
namespace detail {

template <typename DataT>
inline constexpr bool is_fused_1nn_cutile_data_v =
  std::is_same_v<DataT, float> || std::is_same_v<DataT, half>;

template <typename OutT, typename IdxT, typename DataT>
inline constexpr bool is_fused_1nn_kvp_output_v =
  is_fused_1nn_cutile_data_v<DataT> && (std::is_same_v<OutT, raft::KeyValuePair<IdxT, float>> ||
                                        std::is_same_v<OutT, raft::KeyValuePair<IdxT, DataT>>);

template <typename OutT, typename IdxT, typename DataT>
concept Fused1nnKvpOutput = is_fused_1nn_kvp_output_v<OutT, IdxT, DataT>;

#if CUVS_CUTILE_ENABLED
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
#else
template <typename DataT, typename OutT, typename IdxT>
  requires Fused1nnKvpOutput<OutT, IdxT, DataT>
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
#endif

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
