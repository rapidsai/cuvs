/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ivf_pq_serialize.cuh"
#include <cuvs/neighbors/ivf_pq.hpp>
#include <sstream>

namespace cuvs::neighbors::ivf_pq {

void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::ivf_pq::index<int64_t>& index)
{
  cuvs::neighbors::ivf_pq::detail::serialize(handle, filename, index);
}

void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::ivf_pq::index<int64_t>& index)
{
  cuvs::neighbors::ivf_pq::detail::serialize(handle, os, index);
}
}  // namespace cuvs::neighbors::ivf_pq
