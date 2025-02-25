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

#include "cuvs/neighbors/common.hpp"
#include <bang.h>
#include <cstdint>

namespace cuvs::neighbors::experimental::bang {

/**
 * @defgroup bang_cpp_index_params bang index wrapper params
 * @{
 */
struct search_params : cuvs::neighbors::search_params {
  int worklist_length = 0;  // worklist length for greedy search
};

template <typename T>
struct index : cuvs::neighbors::index {
  index(raft::resources const& res,
        const std::string& disk_index_path,
        cuvs::distance::DistanceType metric);

  BANGSearch<T> bang_instance;
  /** Distance metric used for clustering. */
  [[nodiscard]] constexpr inline auto metric() const noexcept -> cuvs::distance::DistanceType
  {
    return metric_;
  }

 private:
  cuvs::distance::DistanceType metric_;
};

template <typename T>
void search(raft::resources const& handle,
            const search_params& params,
            const index<T>& index,
            raft::device_matrix_view<int, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);
/**
 * @}
 */
}  // namespace cuvs::neighbors::experimental::bang
