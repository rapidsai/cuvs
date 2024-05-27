/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuvs/selection/select_k.hpp>
#include <raft/matrix/select_k.cuh>

namespace cuvs::selection {

template <typename T, typename IdxT>
void select_k(raft::resources const& handle,
              raft::device_matrix_view<const T, int64_t, raft::row_major> in_val,
              std::optional<raft::device_matrix_view<const IdxT, int64_t, raft::row_major>> in_idx,
              raft::device_matrix_view<T, int64_t, raft::row_major> out_val,
              raft::device_matrix_view<IdxT, int64_t, raft::row_major> out_idx,
              bool select_min,
              bool sorted,
              SelectAlgo algo)
{
  // just delegate implementation to raft - the primary benefit here is to have
  // instantiations only compiled once in cuvs
  raft::matrix::select_k(handle, in_val, in_idx, out_val, out_idx, select_min, sorted, algo);
}
}  // namespace cuvs::selection
