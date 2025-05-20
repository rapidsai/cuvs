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

#include "vamana.cuh"

namespace cuvs::neighbors::vamana {

#define RAFT_INST_VAMANA_CODEBOOKS(T)                                       \
  auto get_codebooks(const std::string& codebook_prefix, const int dim)     \
    -> cuvs::neighbors::vamana::index_params::codebook_params<T>            \
  {                                                                         \
    return cuvs::neighbors::vamana::get_codebooks<T>(codebook_prefix, dim); \
  }

RAFT_INST_VAMANA_CODEBOOKS(float);

#undef RAFT_INST_VAMANA_CODEBOOKS

}  // namespace cuvs::neighbors::vamana
