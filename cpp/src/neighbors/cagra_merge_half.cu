/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cagra.cuh"
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {

#define RAFT_INST_CAGRA_MERGE(T, IdxT)                                      \
  auto merge(raft::resources const& handle,                                 \
             const cuvs::neighbors::cagra::merge_params& params,            \
             std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices) \
    -> cuvs::neighbors::cagra::index<T, IdxT>                               \
  {                                                                         \
    return cuvs::neighbors::cagra::merge<T, IdxT>(handle, params, indices); \
  }

RAFT_INST_CAGRA_MERGE(half, uint32_t);

#undef RAFT_INST_CAGRA_MERGE

}  // namespace cuvs::neighbors::cagra
