/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdspan.hpp>
#include <variant>

namespace cuvs::neighbors::ivf_pq::detail {

template <typename IdxT>
void unpack_contiguous_list_data(
  uint8_t* codes,
  raft::device_mdspan<const uint8_t,
                      typename list_spec<uint32_t, IdxT>::list_extents,
                      raft::row_major> list_data,
  uint32_t n_rows,
  uint32_t pq_dim,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t pq_bits,
  rmm::cuda_stream_view stream);

template <typename IdxT>
void unpack_contiguous_list_data(raft::resources const& res,
                                 const index<IdxT>& index,
                                 uint8_t* out_codes,
                                 uint32_t n_rows,
                                 uint32_t label,
                                 std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  unpack_contiguous_list_data<IdxT>(out_codes,
                                    index.lists()[label]->data.view(),
                                    n_rows,
                                    index.pq_dim(),
                                    offset_or_indices,
                                    index.pq_bits(),
                                    raft::resource::get_cuda_stream(res));
};

template <typename IdxT>
void pack_contiguous_list_data(
  raft::device_mdspan<uint8_t, typename list_spec<uint32_t, IdxT>::list_extents, raft::row_major>
    list_data,
  const uint8_t* codes,
  uint32_t n_rows,
  uint32_t pq_dim,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t pq_bits,
  rmm::cuda_stream_view stream);

template <typename IdxT>
void pack_contiguous_list_data(raft::resources const& res,
                               index<IdxT>* index,
                               const uint8_t* new_codes,
                               uint32_t n_rows,
                               uint32_t label,
                               std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  pack_contiguous_list_data<IdxT>(index->lists()[label]->data.view(),
                                  new_codes,
                                  n_rows,
                                  index->pq_dim(),
                                  offset_or_indices,
                                  index->pq_bits(),
                                  raft::resource::get_cuda_stream(res));
};

}  // namespace cuvs::neighbors::ivf_pq::detail
