/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include "detail/vamana/vamana_serialize.cuh"

namespace cuvs::neighbors::vamana {

/**
 * @defgroup VAMANA graph serialize/derserialize
 * @{
 */

#define CUVS_INST_VAMANA_SERIALIZE(DTYPE)                                       \
  void serialize(raft::resources const& handle,                                 \
                 const std::string& file_prefix,                                \
                 const cuvs::neighbors::vamana::index<DTYPE, uint32_t>& index_, \
                 bool include_dataset)                                          \
  {                                                                             \
    cuvs::neighbors::vamana::detail::serialize<DTYPE, uint32_t>(                \
      handle, file_prefix, index_, include_dataset);                            \
  };

/** @} */  // end group vamana

}  // namespace cuvs::neighbors::vamana
