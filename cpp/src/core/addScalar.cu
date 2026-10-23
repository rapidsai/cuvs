/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "map_launchers.hpp"

#include <cuvs/core/cuda_fp16.hpp>
#include <raft/linalg/add.cuh>

namespace cuvs::detail {

template <typename InT, typename OutT, typename IdxType>
void addScalar(OutT* out, const InT* in, const InT scalar, IdxType len, cudaStream_t stream)

{
  return raft::linalg::addScalar(out, in, scalar, len, stream);
}

}  // namespace cuvs::detail
