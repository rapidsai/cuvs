/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdspan.hpp>
#include <variant>

namespace cuvs::neighbors::ivf_pq::detail {

// Interleaved layout versions
void unpack_list_data_interleaved(
  raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> codes,
  raft::device_mdspan<const uint8_t,
                      list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                      raft::row_major> list_data,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t pq_bits,
  rmm::cuda_stream_view stream);

void pack_list_data_interleaved(
  raft::device_mdspan<uint8_t,
                      list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                      raft::row_major> list_data,
  raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t pq_bits,
  rmm::cuda_stream_view stream);

// Flat layout versions
void unpack_list_data_flat(raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> codes,
                           const uint8_t* list_data,
                           std::variant<uint32_t, const uint32_t*> offset_or_indices,
                           uint32_t pq_bits,
                           rmm::cuda_stream_view stream);

void pack_list_data_flat(uint8_t* list_data,
                         raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
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
  auto stream = raft::resource::get_cuda_stream(res);
  if (index.codes_layout() == list_layout::FLAT) {
    unpack_list_data_flat(out_codes,
                          index.lists()[label]->data.data_handle(),
                          offset_or_indices,
                          index.pq_bits(),
                          stream);
  } else {
    unpack_list_data_interleaved(
      out_codes, index.lists()[label]->data.view(), offset_or_indices, index.pq_bits(), stream);
  }
}

template <typename IdxT>
void pack_list_data(raft::resources const& res,
                    index<IdxT>* index,
                    raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> new_codes,
                    uint32_t label,
                    std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  auto stream = raft::resource::get_cuda_stream(res);
  if (index->codes_layout() == list_layout::FLAT) {
    pack_list_data_flat(index->lists()[label]->data.data_handle(),
                        new_codes,
                        offset_or_indices,
                        index->pq_bits(),
                        stream);
  } else {
    pack_list_data_interleaved(
      index->lists()[label]->data.view(), new_codes, offset_or_indices, index->pq_bits(), stream);
  }
}

}  // namespace cuvs::neighbors::ivf_pq::detail
