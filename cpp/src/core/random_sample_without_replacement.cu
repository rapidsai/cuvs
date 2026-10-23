/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <core/cub_ptx_query.hpp>
#include <core/random_launchers.hpp>
#include <raft/random/rng.cuh>

namespace cuvs::detail::random {

template <typename T, typename Weight, typename Index>
void sampleWithoutReplacement(raft::resources const& res,
                              raft::random::RngState& state,
                              T* out,
                              Index* out_indices,
                              const T* input,
                              const Weight* weights,
                              Index sampled_len,
                              Index len)
{
  raft::random::sampleWithoutReplacement(
    res, state, out, out_indices, input, weights, sampled_len, len);
}

}  // namespace cuvs::detail::random

template void cuvs::detail::random::sampleWithoutReplacement<int64_t, float, int64_t>(
  raft::resources const&,
  raft::random::RngState&,
  int64_t*,
  int64_t*,
  const int64_t*,
  const float*,
  int64_t,
  int64_t);
