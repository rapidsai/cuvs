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

#include "cagra_serialize.cuh"
#include <raft/core/device_resources.hpp>

#include <cuvs/neighbors/cagra.hpp>

#include <cuda_fp16.h>

#include <sstream>
#include <string>

namespace cuvs::neighbors::cagra {

#define RAFT_INST_CAGRA_SERIALIZE(DTYPE)                                                          \
  void serialize_file(raft::resources const& handle,                                              \
                      const std::string& filename,                                                \
                      const cuvs::neighbors::cagra::index<DTYPE, uint32_t>& index,                \
                      bool include_dataset)                                                       \
  {                                                                                               \
    cuvs::neighbors::cagra::serialize<DTYPE, uint32_t>(handle, filename, index, include_dataset); \
  };                                                                                              \
                                                                                                  \
  void deserialize_file(raft::resources const& handle,                                            \
                        const std::string& filename,                                              \
                        cuvs::neighbors::cagra::index<DTYPE, uint32_t>* index)                    \
  {                                                                                               \
    if (!index) { RAFT_FAIL("Invalid index pointer"); }                                           \
    *index = cuvs::neighbors::cagra::deserialize<DTYPE, uint32_t>(handle, filename);              \
  };                                                                                              \
  void serialize(raft::resources const& handle,                                                   \
                 std::string& str,                                                                \
                 const cuvs::neighbors::cagra::index<DTYPE, uint32_t>& index,                     \
                 bool include_dataset)                                                            \
  {                                                                                               \
    std::stringstream os;                                                                         \
    cuvs::neighbors::cagra::serialize<DTYPE, uint32_t>(handle, os, index, include_dataset);       \
    str = os.str();                                                                               \
  }                                                                                               \
                                                                                                  \
  void serialize_to_hnswlib_file(raft::resources const& handle,                                   \
                                 const std::string& filename,                                     \
                                 const cuvs::neighbors::cagra::index<DTYPE, uint32_t>& index)     \
  {                                                                                               \
    cuvs::neighbors::cagra::serialize_to_hnswlib<DTYPE, uint32_t>(handle, filename, index);       \
  };                                                                                              \
  void serialize_to_hnswlib(raft::resources const& handle,                                        \
                            std::string& str,                                                     \
                            const cuvs::neighbors::cagra::index<DTYPE, uint32_t>& index)          \
  {                                                                                               \
    std::stringstream os;                                                                         \
    cuvs::neighbors::cagra::serialize_to_hnswlib<DTYPE, uint32_t>(handle, os, index);             \
    str = os.str();                                                                               \
  }                                                                                               \
                                                                                                  \
  void deserialize(raft::resources const& handle,                                                 \
                   const std::string& str,                                                        \
                   cuvs::neighbors::cagra::index<DTYPE, uint32_t>* index)                         \
  {                                                                                               \
    std::istringstream is(str);                                                                   \
    if (!index) { RAFT_FAIL("Invalid index pointer"); }                                           \
    *index = cuvs::neighbors::cagra::deserialize<DTYPE, uint32_t>(handle, is);                    \
  }

RAFT_INST_CAGRA_SERIALIZE(uint8_t);

#undef RAFT_INST_CAGRA_SERIALIZE
}  // namespace cuvs::neighbors::cagra
