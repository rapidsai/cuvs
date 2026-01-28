/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/operators.hpp>

namespace cuvs::neighbors::ivf_flat::detail {

template <typename T>
__device__ T post_process(T val)
{
  return raft::identity_op{}(val);
}

}  // namespace cuvs::neighbors::ivf_flat::detail
