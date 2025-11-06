/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ivf_pq_contiguous_list_data_impl.cuh"
#include <cuvs/neighbors/ivf_pq.hpp>

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
  rmm::cuda_stream_view stream)
{
  unpack_contiguous_list_data_impl<IdxT>(
    codes, list_data, n_rows, pq_dim, offset_or_indices, pq_bits, stream);
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
  rmm::cuda_stream_view stream)
{
  pack_contiguous_list_data_impl<IdxT>(
    list_data, codes, n_rows, pq_dim, offset_or_indices, pq_bits, stream);
};

// Explicit instantiations
template void unpack_contiguous_list_data<uint32_t>(
  uint8_t*,
  raft::device_mdspan<const uint8_t,
                      typename list_spec<uint32_t, uint32_t>::list_extents,
                      raft::row_major>,
  uint32_t,
  uint32_t,
  std::variant<uint32_t, const uint32_t*>,
  uint32_t,
  rmm::cuda_stream_view);

template void unpack_contiguous_list_data<int64_t>(
  uint8_t*,
  raft::device_mdspan<const uint8_t,
                      typename list_spec<uint32_t, int64_t>::list_extents,
                      raft::row_major>,
  uint32_t,
  uint32_t,
  std::variant<uint32_t, const uint32_t*>,
  uint32_t,
  rmm::cuda_stream_view);

template void pack_contiguous_list_data<uint32_t>(
  raft::
    device_mdspan<uint8_t, typename list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>,
  const uint8_t*,
  uint32_t,
  uint32_t,
  std::variant<uint32_t, const uint32_t*>,
  uint32_t,
  rmm::cuda_stream_view);

template void pack_contiguous_list_data<int64_t>(
  raft::
    device_mdspan<uint8_t, typename list_spec<uint32_t, int64_t>::list_extents, raft::row_major>,
  const uint8_t*,
  uint32_t,
  uint32_t,
  std::variant<uint32_t, const uint32_t*>,
  uint32_t,
  rmm::cuda_stream_view);
};  // namespace cuvs::neighbors::ivf_pq::detail
