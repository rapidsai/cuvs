/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ivf_pq_process_and_fill_codes_impl.cuh"
#include <cuvs/neighbors/ivf_pq.hpp>

#define instantiate_cuvs_neighbors_ivf_pq_detail_process_and_fill_codes(IdxT)                \
  template void cuvs::neighbors::ivf_pq::detail::launch_process_and_fill_codes_kernel<IdxT>( \
    raft::resources const& handle,                                                           \
    cuvs::neighbors::ivf_pq::index<IdxT>& index,                                             \
    raft::device_matrix_view<float, IdxT> new_vectors_residual,                              \
    std::variant<IdxT, const IdxT*> src_offset_or_indices,                                   \
    const uint32_t* new_labels,                                                              \
    IdxT n_rows);

instantiate_cuvs_neighbors_ivf_pq_detail_process_and_fill_codes(int64_t);
