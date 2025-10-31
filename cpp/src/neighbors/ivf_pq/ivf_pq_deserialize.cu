/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ivf_pq_serialize.cuh"
#include <cuvs/neighbors/ivf_pq.hpp>
#include <sstream>

namespace cuvs::neighbors::ivf_pq {

void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_pq::index<int64_t>* index)
{
  if (!index) { RAFT_FAIL("Invalid index pointer"); }
  *index = cuvs::neighbors::ivf_pq::detail::deserialize<int64_t>(handle, filename);
}

void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::ivf_pq::index<int64_t>* index)
{
  if (!index) { RAFT_FAIL("Invalid index pointer"); }
  *index = cuvs::neighbors::ivf_pq::detail::deserialize<int64_t>(handle, is);
}
}  // namespace cuvs::neighbors::ivf_pq
