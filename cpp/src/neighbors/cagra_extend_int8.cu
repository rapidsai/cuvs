/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cagra.cuh"
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {

#define RAFT_INST_CAGRA_EXTEND(T, IdxT)                                                         \
  void extend(raft::resources const& handle,                                                    \
              const cagra::extend_params& params,                                               \
              raft::device_matrix_view<const T, int64_t, raft::row_major> additional_dataset,   \
              cuvs::neighbors::cagra::index<T, IdxT>& idx,                                      \
              std::optional<raft::device_matrix_view<T, int64_t, raft::layout_stride>> ndv,     \
              std::optional<raft::device_matrix_view<IdxT, int64_t>> ngv)                       \
  {                                                                                             \
    cuvs::neighbors::cagra::extend<T, IdxT>(handle, additional_dataset, idx, params, ndv, ngv); \
  }                                                                                             \
                                                                                                \
  void extend(raft::resources const& handle,                                                    \
              const cagra::extend_params& params,                                               \
              raft::host_matrix_view<const T, int64_t, raft::row_major> additional_dataset,     \
              cuvs::neighbors::cagra::index<T, IdxT>& idx,                                      \
              std::optional<raft::device_matrix_view<T, int64_t, raft::layout_stride>> ndv,     \
              std::optional<raft::device_matrix_view<IdxT, int64_t>> ngv)                       \
  {                                                                                             \
    cuvs::neighbors::cagra::extend<T, IdxT>(handle, additional_dataset, idx, params, ndv, ngv); \
  }

RAFT_INST_CAGRA_EXTEND(int8_t, uint32_t);

#undef RAFT_INST_CAGRA_EXTEND

}  // namespace cuvs::neighbors::cagra
