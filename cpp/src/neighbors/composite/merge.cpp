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
#include <cstddef>
#include <cstdint>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/composite/merge.hpp>
#include <cuvs/neighbors/index_base.hpp>
#include <cuvs/neighbors/index_wrappers.hpp>
#include <memory>
#include <raft/core/error.hpp>
#include <raft/core/resources.hpp>
#include <vector>

namespace cuvs::neighbors::composite {

/**
 * @brief Merge multiple indices into a single composite index.
 *
 * This function provides polymorphic merge capability for different index types.
 * It delegates to the first index's merge method, which handles the actual merging
 * logic based on the index type and merge parameters.
 *
 * @tparam T Data element type
 * @tparam IdxT Index type for vector indices
 * @tparam OutputIdxT Output index type
 * @param[in] handle RAFT resources for executing operations
 * @param[in] params Merge parameters containing strategy and algorithm-specific settings
 * @param[in] indices Vector of IndexWrapper pointers to merge
 * @return Shared pointer to merged composite index
 */
template <typename T, typename IdxT, typename OutputIdxT>
std::shared_ptr<cuvs::neighbors::IndexBase<T, IdxT, OutputIdxT>> merge(
  const raft::resources& handle,
  const cuvs::neighbors::merge_params& params,
  std::vector<std::shared_ptr<cuvs::neighbors::IndexWrapper<T, IdxT, OutputIdxT>>>& indices)
{
  if (indices.empty()) { RAFT_FAIL("Cannot merge empty indices vector"); }

  if (indices.size() == 1) { return indices[0]; }

  auto first_index = indices[0];
  std::vector<std::shared_ptr<cuvs::neighbors::IndexBase<T, IdxT, OutputIdxT>>> other_indices;
  for (std::size_t i = 1; i < indices.size(); ++i) {
    other_indices.push_back(indices[i]);
  }

  // Delegate to the first index's merge method
  return first_index->merge(handle, params, other_indices);
}

#define INSTANTIATE(T, IdxT, OutputIdxT)                                    \
  template std::shared_ptr<cuvs::neighbors::IndexBase<T, IdxT, OutputIdxT>> \
  merge<T, IdxT, OutputIdxT>(                                               \
    const raft::resources&,                                                 \
    const cuvs::neighbors::merge_params&,                                   \
    std::vector<std::shared_ptr<cuvs::neighbors::IndexWrapper<T, IdxT, OutputIdxT>>>&);

INSTANTIATE(float, uint32_t, uint32_t);
INSTANTIATE(half, uint32_t, uint32_t);
INSTANTIATE(int8_t, uint32_t, uint32_t);
INSTANTIATE(uint8_t, uint32_t, uint32_t);

INSTANTIATE(float, uint32_t, int64_t);
INSTANTIATE(half, uint32_t, int64_t);
INSTANTIATE(int8_t, uint32_t, int64_t);
INSTANTIATE(uint8_t, uint32_t, int64_t);

#undef INSTANTIATE

}  // namespace cuvs::neighbors::composite
