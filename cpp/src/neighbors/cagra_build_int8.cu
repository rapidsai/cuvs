/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cagra.cuh"
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {

#define RAFT_INST_CAGRA_BUILD(T, IdxT)                                            \
  auto build(raft::resources const& handle,                                       \
             const cuvs::neighbors::cagra::index_params& params,                  \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset) \
    -> cuvs::neighbors::cagra::index<T, IdxT>                                     \
  {                                                                               \
    return cuvs::neighbors::cagra::build<T, IdxT>(handle, params, dataset);       \
  }                                                                               \
                                                                                  \
  auto build(raft::resources const& handle,                                       \
             const cuvs::neighbors::cagra::index_params& params,                  \
             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)   \
    -> cuvs::neighbors::cagra::index<T, IdxT>                                     \
  {                                                                               \
    return cuvs::neighbors::cagra::build<T, IdxT>(handle, params, dataset);       \
  }

RAFT_INST_CAGRA_BUILD(int8_t, uint32_t);

#undef RAFT_INST_CAGRA_BUILD

}  // namespace cuvs::neighbors::cagra
