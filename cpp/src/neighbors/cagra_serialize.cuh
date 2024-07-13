/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "detail/cagra/cagra_serialize.cuh"

namespace cuvs::neighbors::cagra {

#define CUVS_INST_CAGRA_SERIALIZE(DTYPE)                                                      \
  void serialize(raft::resources const& handle,                                               \
                 const std::string& filename,                                                 \
                 const cuvs::neighbors::cagra::index<DTYPE, uint32_t>& index,                 \
                 bool include_dataset)                                                        \
  {                                                                                           \
    cuvs::neighbors::cagra::detail::serialize<DTYPE, uint32_t>(                               \
      handle, filename, index, include_dataset);                                              \
  };                                                                                          \
                                                                                              \
  void deserialize(raft::resources const& handle,                                             \
                   const std::string& filename,                                               \
                   cuvs::neighbors::cagra::index<DTYPE, uint32_t>* index)                     \
  {                                                                                           \
    cuvs::neighbors::cagra::detail::deserialize<DTYPE, uint32_t>(handle, filename, index);    \
  };                                                                                          \
  void serialize(raft::resources const& handle,                                               \
                 std::ostream& os,                                                            \
                 const cuvs::neighbors::cagra::index<DTYPE, uint32_t>& index,                 \
                 bool include_dataset)                                                        \
  {                                                                                           \
    cuvs::neighbors::cagra::detail::serialize<DTYPE, uint32_t>(                               \
      handle, os, index, include_dataset);                                                    \
  }                                                                                           \
                                                                                              \
  void deserialize(raft::resources const& handle,                                             \
                   std::istream& is,                                                          \
                   cuvs::neighbors::cagra::index<DTYPE, uint32_t>* index)                     \
  {                                                                                           \
    cuvs::neighbors::cagra::detail::deserialize<DTYPE, uint32_t>(handle, is, index);          \
  }                                                                                           \
                                                                                              \
  void serialize_to_hnswlib(raft::resources const& handle,                                    \
                            std::ostream& os,                                                 \
                            const cuvs::neighbors::cagra::index<DTYPE, uint32_t>& index)      \
  {                                                                                           \
    cuvs::neighbors::cagra::detail::serialize_to_hnswlib<DTYPE, uint32_t>(handle, os, index); \
  }                                                                                           \
                                                                                              \
  void serialize_to_hnswlib(raft::resources const& handle,                                    \
                            const std::string& filename,                                      \
                            const cuvs::neighbors::cagra::index<DTYPE, uint32_t>& index)      \
  {                                                                                           \
    cuvs::neighbors::cagra::detail::serialize_to_hnswlib<DTYPE, uint32_t>(                    \
      handle, filename, index);                                                               \
  }

}  // namespace cuvs::neighbors::cagra
