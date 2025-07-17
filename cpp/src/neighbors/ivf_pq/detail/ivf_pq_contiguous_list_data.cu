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

#include "../ivf_pq_contiguous_list_data_impl.cuh"
#include <cuvs/neighbors/ivf_pq.hpp>

#define instantiate_cuvs_neighbors_ivf_pq_detail_contiguous_list_data(IdxT)                    \
  namespace cuvs::neighbors::ivf_pq::detail {                                                  \
  void unpack_contiguous_list_data(                                                            \
    uint8_t* codes,                                                                            \
    raft::device_mdspan<const uint8_t,                                                         \
                        list_spec<uint32_t, uint32_t>::list_extents,                           \
                        raft::row_major> list_data,                                            \
    uint32_t n_rows,                                                                           \
    uint32_t pq_dim,                                                                           \
    std::variant<uint32_t, const uint32_t*> offset_or_indices,                                 \
    uint32_t pq_bits,                                                                          \
    rmm::cuda_stream_view stream)                                                              \
  {                                                                                            \
    unpack_contiguous_list_data_impl(                                                          \
      codes, list_data, n_rows, pq_dim, offset_or_indices, pq_bits, stream);                   \
  };                                                                                           \
                                                                                               \
  void pack_contiguous_list_data(                                                              \
    raft::device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major> \
      list_data,                                                                               \
    const uint8_t* codes,                                                                      \
    uint32_t n_rows,                                                                           \
    uint32_t pq_dim,                                                                           \
    std::variant<uint32_t, const uint32_t*> offset_or_indices,                                 \
    uint32_t pq_bits,                                                                          \
    rmm::cuda_stream_view stream)                                                              \
  {                                                                                            \
    pack_contiguous_list_data_impl(                                                            \
      list_data, codes, n_rows, pq_dim, offset_or_indices, pq_bits, stream);                   \
  };                                                                                           \
  };

instantiate_cuvs_neighbors_ivf_pq_detail_contiguous_list_data(int64_t);
