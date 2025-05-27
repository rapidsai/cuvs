/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "cagra.cuh"
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {

#define CUVS_INST_CAGRA_SEARCH(T, IdxT, OutputIdxT)                                     \
  void search(raft::resources const& handle,                                            \
              cuvs::neighbors::cagra::search_params const& params,                      \
              const cuvs::neighbors::cagra::index<T, IdxT>& index,                      \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,      \
              raft::device_matrix_view<OutputIdxT, int64_t, raft::row_major> neighbors, \
              raft::device_matrix_view<float, int64_t, raft::row_major> distances,      \
              const cuvs::neighbors::filtering::base_filter& sample_filter)             \
  {                                                                                     \
    cuvs::neighbors::cagra::search<T, IdxT, OutputIdxT>(                                \
      handle, params, index, queries, neighbors, distances, sample_filter);             \
  }

CUVS_INST_CAGRA_SEARCH(uint8_t, uint32_t, uint32_t);
CUVS_INST_CAGRA_SEARCH(uint8_t, uint32_t, int64_t);

#undef CUVS_INST_CAGRA_SEARCH

}  // namespace cuvs::neighbors::cagra
