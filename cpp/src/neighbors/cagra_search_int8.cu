/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

CUVS_INST_CAGRA_SEARCH(int8_t, uint32_t, uint32_t);
CUVS_INST_CAGRA_SEARCH(int8_t, uint32_t, int64_t);

#undef CUVS_INST_CAGRA_SEARCH

}  // namespace cuvs::neighbors::cagra
