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

#pragma once

#include "../index_base.hpp"
#include "../index_wrappers.hpp"
#include <cuvs/neighbors/cagra.hpp>
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
template <typename T, typename IdxT, typename OutputIdxT = IdxT>
std::shared_ptr<IndexBase<T, IdxT, OutputIdxT>> merge(
  const raft::resources& handle,
  const cuvs::neighbors::merge_params& params,
  std::vector<std::shared_ptr<cuvs::neighbors::IndexWrapper<T, IdxT, OutputIdxT>>>& indices);

}  // namespace cuvs::neighbors::composite
