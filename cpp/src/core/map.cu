/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../neighbors/detail/ann_utils.cuh"
#include "map.hpp"

namespace cuvs::detail {

template <bool PassOffset, typename OutT, typename IdxT, typename Func, typename... InTs>
void map_impl(cuda::stream_ref stream, OutT* out_ptr, IdxT len, Func f, const InTs*... in_ptrs)
{
  raft::linalg::detail::map<PassOffset>(stream, out_ptr, len, f, in_ptrs...);
}

// EXPLICIT_HOST_INSTANTIATIONS

}  // namespace cuvs::detail
