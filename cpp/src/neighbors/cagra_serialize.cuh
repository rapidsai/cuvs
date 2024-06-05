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
/**
 * Write the CAGRA built index as a base layer HNSW index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra_serialize.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = raft::cagra::build(...);`
 * raft::cagra::serialize_to_hnswlib(handle, os, index);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index CAGRA index
 *
 */
template <typename T, typename IdxT>
void serialize_to_hnswlib(raft::resources const& handle,
                          std::ostream& os,
                          const cuvs::neighbors::cagra::index<T, IdxT>& index)
{
  detail::serialize_to_hnswlib<T, IdxT>(handle, os, index);
}

/**
 * Save a CAGRA build index in hnswlib base-layer-only serialized format
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra_serialize.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = raft::cagra::build(...);`
 * raft::cagra::serialize_to_hnswlib(handle, filename, index);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 *
 */
template <typename T, typename IdxT>
void serialize_to_hnswlib(raft::resources const& handle,
                          const std::string& filename,
                          const cuvs::neighbors::cagra::index<T, IdxT>& index)
{
  detail::serialize_to_hnswlib<T, IdxT>(handle, filename, index);
}

#define CUVS_INST_CAGRA_SERIALIZE(DTYPE)                                                   \
  void serialize(raft::resources const& handle,                                            \
                 const std::string& filename,                                              \
                 const cuvs::neighbors::cagra::index<DTYPE, uint32_t>& index,              \
                 bool include_dataset)                                                     \
  {                                                                                        \
    cuvs::neighbors::cagra::detail::serialize<DTYPE, uint32_t>(                            \
      handle, filename, index, include_dataset);                                           \
  };                                                                                       \
                                                                                           \
  void deserialize(raft::resources const& handle,                                          \
                   const std::string& filename,                                            \
                   cuvs::neighbors::cagra::index<DTYPE, uint32_t>* index)                  \
  {                                                                                        \
    cuvs::neighbors::cagra::detail::deserialize<DTYPE, uint32_t>(handle, filename, index); \
  };                                                                                       \
  void serialize(raft::resources const& handle,                                            \
                 std::ostream& os,                                                         \
                 const cuvs::neighbors::cagra::index<DTYPE, uint32_t>& index,              \
                 bool include_dataset)                                                     \
  {                                                                                        \
    cuvs::neighbors::cagra::detail::serialize<DTYPE, uint32_t>(                            \
      handle, os, index, include_dataset);                                                 \
  }                                                                                        \
                                                                                           \
  void deserialize(raft::resources const& handle,                                          \
                   std::istream& is,                                                       \
                   cuvs::neighbors::cagra::index<DTYPE, uint32_t>* index)                  \
  {                                                                                        \
    cuvs::neighbors::cagra::detail::deserialize<DTYPE, uint32_t>(handle, is, index);       \
  }

}  // namespace cuvs::neighbors::cagra
