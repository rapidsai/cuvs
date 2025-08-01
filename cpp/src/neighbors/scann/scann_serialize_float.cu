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

#include "scann.cuh"
#include <cuvs/neighbors/scann.hpp>

namespace cuvs::neighbors::experimental::scann {

/**
 * @defgroup ScaNN graph serialize/derserialize
 * @{
 */

#define CUVS_INST_SCANN_SERIALIZE(DTYPE, IdxT)                                           \
  void serialize(raft::resources const& handle,                                          \
                 const std::string& file_prefix,                                         \
                 const cuvs::neighbors::experimental::scann::index<DTYPE, IdxT>& index_) \
  {                                                                                      \
    cuvs::neighbors::experimental::scann::detail::serialize<DTYPE, IdxT>(                \
      handle, file_prefix, index_);                                                      \
  };

CUVS_INST_SCANN_SERIALIZE(float, int64_t);

/** @} */  // end group scann

}  // namespace cuvs::neighbors::experimental::scann
