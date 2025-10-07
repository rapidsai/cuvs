/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "../../sample_filter.cuh"

namespace cuvs::neighbors::ivf_flat::detail {

template <typename index_t>
__device__ bool sample_filter(index_t* const* const inds_ptrs,
                              const uint32_t query_ix,
                              const uint32_t cluster_ix,
                              const uint32_t sample_ix,
                              uint32_t* bitset_ptr,
                              index_t bitset_len,
                              index_t original_nbits)
{
  auto bitset_view =
    raft::core::bitset_view<uint32_t, index_t>{bitset_ptr, bitset_len, original_nbits};
  auto bitset_filter = cuvs::neighbors::filtering::bitset_filter<uint32_t, index_t>{bitset_view};
  auto ivf_to_sample_filter = cuvs::neighbors::filtering::
    ivf_to_sample_filter<index_t, cuvs::neighbors::filtering::bitset_filter<uint32_t, index_t>>{
      inds_ptrs, bitset_filter};
  return ivf_to_sample_filter(query_ix, cluster_ix, sample_ix);
}

}  // namespace cuvs::neighbors::ivf_flat::detail
