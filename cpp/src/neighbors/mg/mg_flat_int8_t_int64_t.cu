/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

/*
 * NOTE: this file is generated by generate_mg.py
 *
 * Make changes there and run in this directory:
 *
 * > python generate_mg.py
 *
 */

#include "mg.cuh"

#define CUVS_INST_MG_FLAT(T, IdxT)                                                               \
  namespace cuvs::neighbors::ivf_flat {                                                          \
  using namespace cuvs::neighbors::mg;                                                           \
                                                                                                 \
  cuvs::neighbors::mg::index<ivf_flat::index<T, IdxT>, T, IdxT> build(                           \
    const raft::resources& res,                                                                  \
    const mg::index_params<ivf_flat::index_params>& index_params,                                \
    raft::host_matrix_view<const T, int64_t, row_major> index_dataset)                           \
  {                                                                                              \
    cuvs::neighbors::mg::index<ivf_flat::index<T, IdxT>, T, IdxT> index(res, index_params.mode); \
    cuvs::neighbors::mg::detail::build(                                                          \
      res,                                                                                       \
      index,                                                                                     \
      static_cast<const cuvs::neighbors::index_params*>(&index_params),                          \
      index_dataset);                                                                            \
    return index;                                                                                \
  }                                                                                              \
                                                                                                 \
  void extend(const raft::resources& res,                                                        \
              cuvs::neighbors::mg::index<ivf_flat::index<T, IdxT>, T, IdxT>& index,              \
              raft::host_matrix_view<const T, int64_t, row_major> new_vectors,                   \
              std::optional<raft::host_vector_view<const IdxT, int64_t>> new_indices)            \
  {                                                                                              \
    cuvs::neighbors::mg::detail::extend(res, index, new_vectors, new_indices);                   \
  }                                                                                              \
                                                                                                 \
  void search(const raft::resources& res,                                                        \
              const cuvs::neighbors::mg::index<ivf_flat::index<T, IdxT>, T, IdxT>& index,        \
              const mg::search_params<ivf_flat::search_params>& search_params,                   \
              raft::host_matrix_view<const T, int64_t, row_major> queries,                       \
              raft::host_matrix_view<IdxT, int64_t, row_major> neighbors,                        \
              raft::host_matrix_view<float, int64_t, row_major> distances)                       \
  {                                                                                              \
    cuvs::neighbors::mg::detail::search(                                                         \
      res,                                                                                       \
      index,                                                                                     \
      static_cast<const cuvs::neighbors::search_params*>(&search_params),                        \
      queries,                                                                                   \
      neighbors,                                                                                 \
      distances);                                                                                \
  }                                                                                              \
                                                                                                 \
  void serialize(const raft::resources& res,                                                     \
                 const cuvs::neighbors::mg::index<ivf_flat::index<T, IdxT>, T, IdxT>& index,     \
                 const std::string& filename)                                                    \
  {                                                                                              \
    cuvs::neighbors::mg::detail::serialize(res, index, filename);                                \
  }                                                                                              \
                                                                                                 \
  template <>                                                                                    \
  cuvs::neighbors::mg::index<ivf_flat::index<T, IdxT>, T, IdxT> deserialize<T, IdxT>(            \
    const raft::resources& res, const std::string& filename)                                     \
  {                                                                                              \
    auto idx = cuvs::neighbors::mg::index<ivf_flat::index<T, IdxT>, T, IdxT>(res, filename);     \
    return idx;                                                                                  \
  }                                                                                              \
                                                                                                 \
  template <>                                                                                    \
  cuvs::neighbors::mg::index<ivf_flat::index<T, IdxT>, T, IdxT> distribute<T, IdxT>(             \
    const raft::resources& res, const std::string& filename)                                     \
  {                                                                                              \
    auto idx = cuvs::neighbors::mg::index<ivf_flat::index<T, IdxT>, T, IdxT>(res, REPLICATED);   \
    cuvs::neighbors::mg::detail::deserialize_and_distribute(res, idx, filename);                 \
    return idx;                                                                                  \
  }                                                                                              \
  }  // namespace cuvs::neighbors::ivf_flat
CUVS_INST_MG_FLAT(int8_t, int64_t);

#undef CUVS_INST_MG_FLAT
