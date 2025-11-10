/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/fused_distance_nn/helper_structs.cuh"
#include <raft/core/resource/cuda_stream.hpp>

namespace cuvs::distance {

/**
 * \defgroup fused_l2_nn Fused 1-nearest neighbors
 * @{
 */

template <typename LabelT, typename DataT>
using KVPMinReduce = detail::KVPMinReduceImpl<LabelT, DataT>;

template <typename LabelT, typename DataT>
using MinAndDistanceReduceOp = detail::MinAndDistanceReduceOpImpl<LabelT, DataT>;

template <typename LabelT, typename DataT>
using MinReduceOp = detail::MinReduceOpImpl<LabelT, DataT>;

/** @} */

/**
 * Initialize array using init value from reduction op
 */
template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT>
void initialize(raft::resources const& handle, OutT* min, IdxT m, DataT maxVal, ReduceOpT redOp)
{
  detail::initialize<DataT, OutT, IdxT, ReduceOpT>(
    min, m, maxVal, redOp, raft::resource::get_cuda_stream(handle));
}

}  // namespace cuvs::distance
