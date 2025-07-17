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

#include "../ivf_pq_process_and_fill_codes_impl.cuh"
#include <cuvs/neighbors/ivf_pq.hpp>

#define instantiate_cuvs_neighbors_ivf_pq_detail_process_and_fill_codes(IdxT)                \
  template void cuvs::neighbors::ivf_pq::detail::launch_process_and_fill_codes_kernel<IdxT>( \
    raft::resources const& handle,                                                           \
    cuvs::neighbors::ivf_pq::index<IdxT>& index,                                             \
    raft::device_matrix_view<float> new_vectors_residual,                                    \
    std::variant<IdxT, const IdxT*> src_offset_or_indices,                                   \
    const uint32_t* new_labels,                                                              \
    IdxT n_rows);

instantiate_cuvs_neighbors_ivf_pq_detail_process_and_fill_codes(int64_t);
