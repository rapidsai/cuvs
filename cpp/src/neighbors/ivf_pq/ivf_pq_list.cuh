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

#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/util/integer_utils.hpp>

namespace cuvs::neighbors::ivf_pq {
template <typename SizeT, typename IdxT>
constexpr list_spec<SizeT, IdxT>::list_spec(uint32_t pq_bits,
                                            uint32_t pq_dim,
                                            bool conservative_memory_allocation)
  : pq_bits(pq_bits),
    pq_dim(pq_dim),
    align_min(kIndexGroupSize),
    align_max(conservative_memory_allocation ? kIndexGroupSize : 1024)
{
}

template <typename SizeT, typename IdxT>
template <typename OtherSizeT>
constexpr list_spec<SizeT, IdxT>::list_spec(const list_spec<OtherSizeT, IdxT>& other_spec)
  : pq_bits{other_spec.pq_bits},
    pq_dim{other_spec.pq_dim},
    align_min{other_spec.align_min},
    align_max{other_spec.align_max}
{
}

template <typename SizeT, typename IdxT>
constexpr typename list_spec<SizeT, IdxT>::list_extents list_spec<SizeT, IdxT>::make_list_extents(
  SizeT n_rows) const
{
  // how many elems of pq_dim fit into one kIndexGroupVecLen-byte chunk
  auto pq_chunk = (kIndexGroupVecLen * 8u) / pq_bits;
  return raft::make_extents<SizeT>(raft::div_rounding_up_safe<SizeT>(n_rows, kIndexGroupSize),
                                   raft::div_rounding_up_safe<SizeT>(pq_dim, pq_chunk),
                                   kIndexGroupSize,
                                   kIndexGroupVecLen);
}
}  // namespace cuvs::neighbors::ivf_pq