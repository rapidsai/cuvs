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

void unpack_list_data(
  raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> codes,
  raft::device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t pq_bits,
  rmm::cuda_stream_view stream);

/** Unpack the list data; see the public interface for the api and usage. */
template <typename IdxT>
void unpack_list_data(raft::resources const& res,
                      const index<IdxT>& index,
                      raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes,
                      uint32_t label,
                      std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  unpack_list_data(out_codes,
                   index.lists()[label]->data.view(),
                   offset_or_indices,
                   index.pq_bits(),
                   raft::resource::get_cuda_stream(res));
}

void pack_list_data(
  raft::device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data,
  raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t pq_bits,
  rmm::cuda_stream_view stream);

template <typename IdxT>
void pack_list_data(raft::resources const& res,
                    index<IdxT>* index,
                    raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> new_codes,
                    uint32_t label,
                    std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  pack_list_data(index->lists()[label]->data.view(),
                 new_codes,
                 offset_or_indices,
                 index->pq_bits(),
                 raft::resource::get_cuda_stream(res));
}

}  // namespace cuvs::neighbors::ivf_pq::detail
