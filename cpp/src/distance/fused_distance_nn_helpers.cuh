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

template <typename label_t, typename DataT>
using kvp_min_reduce = detail::kvp_min_reduce_impl<label_t, DataT>;

template <typename label_t, typename DataT>
using min_and_distance_reduce_op = detail::min_and_distance_reduce_op_impl<label_t, DataT>;

template <typename label_t, typename DataT>
using min_reduce_op = detail::min_reduce_op_impl<label_t, DataT>;

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
