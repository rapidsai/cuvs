/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cagra.cuh"
#include <cuda_fp16.h>
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {

cuvs::neighbors::cagra::index<half, uint32_t> build(
  raft::resources const& handle,
  const cuvs::neighbors::cagra::index_params& params,
  raft::device_matrix_view<const half, int64_t, raft::row_major> dataset)
{
  return cuvs::neighbors::cagra::build<half, uint32_t>(handle, params, dataset);
}

cuvs::neighbors::cagra::index<half, uint32_t> build(
  raft::resources const& handle,
  const cuvs::neighbors::cagra::index_params& params,
  raft::host_matrix_view<const half, int64_t, raft::row_major> dataset)
{
  return cuvs::neighbors::cagra::build<half, uint32_t>(handle, params, dataset);
}

}  // namespace cuvs::neighbors::cagra
