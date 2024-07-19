# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

header = """/*
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
 * NOTE: this file is generated by generate_ann_mg.py
 *
 * Make changes there and run in this directory:
 *
 * > python generate_ann_mg.py
 *
 */

"""

include_macro = """
#include "ann_mg.cuh"
"""

namespace_macro = """
namespace cuvs::neighbors::mg {
"""

footer = """
}  // namespace cuvs::neighbors::mg
"""

flat_macro = """
#define CUVS_INST_ANN_MG_FLAT(T, IdxT)                                                                                    \\
  ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT> build(const raft::resources& handle,                                    \\
                                                        const cuvs::neighbors::mg::nccl_clique& clique,                   \\
                                                        const ivf_flat::mg_index_params& index_params,                    \\
                                                        raft::host_matrix_view<const T, IdxT, row_major> index_dataset)   \\
  {                                                                                                                       \\
    return std::move(cuvs::neighbors::mg::detail::build(handle, clique, index_params, index_dataset));                    \\
  }                                                                                                                       \\
                                                                                                                          \\
  void extend(const raft::resources& handle,                                                                              \\
              const cuvs::neighbors::mg::nccl_clique& clique,                                                             \\
              ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,                                                     \\
              raft::host_matrix_view<const T, IdxT, row_major> new_vectors,                                               \\
              std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices)                                        \\
  {                                                                                                                       \\
    cuvs::neighbors::mg::detail::extend(handle, clique, index, new_vectors, new_indices);                                 \\
  }                                                                                                                       \\
                                                                                                                          \\
  void search(const raft::resources& handle,                                                                              \\
              const cuvs::neighbors::mg::nccl_clique& clique,                                                             \\
              const ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,                                               \\
              const ivf_flat::search_params& search_params,                                                               \\
              raft::host_matrix_view<const T, IdxT, row_major> query_dataset,                                             \\
              raft::host_matrix_view<IdxT, IdxT, row_major> neighbors,                                                    \\
              raft::host_matrix_view<float, IdxT, row_major> distances,                                                   \\
              uint64_t n_rows_per_batch)                                                                                  \\
  {                                                                                                                       \\
    cuvs::neighbors::mg::detail::search(handle, clique, index, search_params, query_dataset,                              \\
                                        neighbors, distances, n_rows_per_batch);                                          \\
  }                                                                                                                       \\
                                                                                                                          \\
  void serialize(const raft::resources& handle,                                                                           \\
                 const cuvs::neighbors::mg::nccl_clique& clique,                                                          \\
                 const ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,                                            \\
                 const std::string& filename)                                                                             \\
  {                                                                                                                       \\
    cuvs::neighbors::mg::detail::serialize(handle, clique, index, filename);                                              \\
  }                                                                                                                       \\
                                                                                                                          \\
  template<>                                                                                                                    \\
  ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT> deserialize_flat<T, IdxT>(const raft::resources& handle,                      \\
                                                                            const cuvs::neighbors::mg::nccl_clique& clique,     \\
                                                                            const std::string& filename)                        \\
  {                                                                                                                             \\
    return std::move(cuvs::neighbors::mg::detail::deserialize_flat<T, IdxT>(handle, clique, filename));                         \\
  }                                                                                                                             \\
                                                                                                                                \\
  template<>                                                                                                                    \\
  ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT> distribute_flat<T, IdxT>(const raft::resources& handle,                       \\
                                                                           const cuvs::neighbors::mg::nccl_clique& clique,      \\
                                                                           const std::string& filename)                         \\
  {                                                                                                                             \\
    return std::move(cuvs::neighbors::mg::detail::distribute_flat<T, IdxT>(handle, clique, filename));                          \\
  }
"""

pq_macro = """
#define CUVS_INST_ANN_MG_PQ(T, IdxT)                                                                                      \\
  ann_mg_index<ivf_pq::index<IdxT>, T, IdxT> build(const raft::resources& handle,                                         \\
                                                   const cuvs::neighbors::mg::nccl_clique& clique,                        \\
                                                   const ivf_pq::mg_index_params& index_params,                           \\
                                                   raft::host_matrix_view<const T, IdxT, row_major> index_dataset)        \\
  {                                                                                                                       \\
    return std::move(cuvs::neighbors::mg::detail::build(handle, clique, index_params, index_dataset));                    \\
  }                                                                                                                       \\
                                                                                                                          \\
  void extend(const raft::resources& handle,                                                                              \\
              const cuvs::neighbors::mg::nccl_clique& clique,                                                             \\
              ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>& index,                                                          \\
              raft::host_matrix_view<const T, IdxT, row_major> new_vectors,                                               \\
              std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices)                                        \\
  {                                                                                                                       \\
    cuvs::neighbors::mg::detail::extend(handle, clique, index, new_vectors, new_indices);                                 \\
  }                                                                                                                       \\
                                                                                                                          \\
  void search(const raft::resources& handle,                                                                              \\
              const cuvs::neighbors::mg::nccl_clique& clique,                                                             \\
              const ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>& index,                                                    \\
              const ivf_pq::search_params& search_params,                                                                 \\
              raft::host_matrix_view<const T, IdxT, row_major> query_dataset,                                             \\
              raft::host_matrix_view<IdxT, IdxT, row_major> neighbors,                                                    \\
              raft::host_matrix_view<float, IdxT, row_major> distances,                                                   \\
              uint64_t n_rows_per_batch)                                                                                  \\
  {                                                                                                                       \\
    cuvs::neighbors::mg::detail::search(handle, clique, index, search_params, query_dataset,                              \\
                                        neighbors, distances, n_rows_per_batch);                                          \\
  }                                                                                                                       \\
                                                                                                                          \\
  void serialize(const raft::resources& handle,                                                                           \\
                 const cuvs::neighbors::mg::nccl_clique& clique,                                                          \\
                 const ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>& index,                                                 \\
                 const std::string& filename)                                                                             \\
  {                                                                                                                       \\
    cuvs::neighbors::mg::detail::serialize(handle, clique, index, filename);                                              \\
  }                                                                                                                       \\
                                                                                                                                \\
  template<>                                                                                                                    \\
  ann_mg_index<ivf_pq::index<IdxT>, T, IdxT> deserialize_pq<T, IdxT>(const raft::resources& handle,                             \\
                                                                     const cuvs::neighbors::mg::nccl_clique& clique,            \\
                                                                     const std::string& filename)                               \\
  {                                                                                                                             \\
    return std::move(cuvs::neighbors::mg::detail::deserialize_pq<T, IdxT>(handle, clique, filename));                           \\
  }                                                                                                                             \\
                                                                                                                                \\
  template<>                                                                                                                    \\
  ann_mg_index<ivf_pq::index<IdxT>, T, IdxT> distribute_pq<T, IdxT>(const raft::resources& handle,                              \\
                                                                    const cuvs::neighbors::mg::nccl_clique& clique,             \\
                                                                    const std::string& filename)                                \\
  {                                                                                                                             \\
    return std::move(cuvs::neighbors::mg::detail::distribute_pq<T, IdxT>(handle, clique, filename));                            \\
  }
"""

cagra_macro = """
#define CUVS_INST_ANN_MG_CAGRA(T, IdxT)                                                                                   \\
  ann_mg_index<cagra::index<T, IdxT>, T, IdxT> build(const raft::resources& handle,                                       \\
                                                     const cuvs::neighbors::mg::nccl_clique& clique,                      \\
                                                     const cagra::mg_index_params& index_params,                          \\
                                                     raft::host_matrix_view<const T, IdxT, row_major> index_dataset)      \\
  {                                                                                                                       \\
    return std::move(cuvs::neighbors::mg::detail::build(handle, clique, index_params, index_dataset));                    \\
  }                                                                                                                       \\
                                                                                                                          \\
  void search(const raft::resources& handle,                                                                              \\
              const cuvs::neighbors::mg::nccl_clique& clique,                                                             \\
              const ann_mg_index<cagra::index<T, IdxT>, T, IdxT>& index,                                                  \\
              const cagra::search_params& search_params,                                                                  \\
              raft::host_matrix_view<const T, IdxT, row_major> query_dataset,                                             \\
              raft::host_matrix_view<IdxT, IdxT, row_major> neighbors,                                                    \\
              raft::host_matrix_view<float, IdxT, row_major> distances,                                                   \\
              uint64_t n_rows_per_batch)                                                                                  \\
  {                                                                                                                       \\
    cuvs::neighbors::mg::detail::search(handle, clique, index, search_params, query_dataset,                              \\
                                        neighbors, distances, n_rows_per_batch);                                          \\
  }                                                                                                                       \\
                                                                                                                          \\
  void serialize(const raft::resources& handle,                                                                           \\
                 const cuvs::neighbors::mg::nccl_clique& clique,                                                          \\
                 const ann_mg_index<cagra::index<T, IdxT>, T, IdxT>& index,                                               \\
                 const std::string& filename)                                                                             \\
  {                                                                                                                       \\
    cuvs::neighbors::mg::detail::serialize(handle, clique, index, filename);                                              \\
  }                                                                                                                       \\
                                                                                                                          \\
  template<>                                                                                                                    \\
  ann_mg_index<cagra::index<T, IdxT>, T, IdxT> deserialize_cagra<T, IdxT>(const raft::resources& handle,                        \\
                                                                          const cuvs::neighbors::mg::nccl_clique& clique,       \\
                                                                          const std::string& filename)                          \\
  {                                                                                                                             \\
    return std::move(cuvs::neighbors::mg::detail::deserialize_cagra<T, IdxT>(handle, clique, filename));                        \\
  }                                                                                                                             \\
                                                                                                                                \\
  template<>                                                                                                                    \\
  ann_mg_index<cagra::index<T, IdxT>, T, IdxT> distribute_cagra<T, IdxT>(const raft::resources& handle,                         \\
                                                                         const cuvs::neighbors::mg::nccl_clique& clique,        \\
                                                                         const std::string& filename)                           \\
  {                                                                                                                             \\
    return std::move(cuvs::neighbors::mg::detail::distribute_cagra<T, IdxT>(handle, clique, filename));                         \\
  }
"""

macros_1 = dict(
    flat=dict(
        include=include_macro,
        definition=flat_macro,
        name="CUVS_INST_ANN_MG_FLAT",
    ),
    pq=dict(
        include=include_macro,
        definition=pq_macro,
        name="CUVS_INST_ANN_MG_PQ",
    ),
)

macros_2 = dict(
    cagra=dict(
        include=include_macro,
        definition=cagra_macro,
        name="CUVS_INST_ANN_MG_CAGRA",
    ),
)

types_1 = dict(
    float_int64_t=("float", "int64_t"),
    int8_t_int64_t=("int8_t", "int64_t"),
    uint8_t_int64_t=("uint8_t", "int64_t"),
)

types_2 = dict(
    float_uint32_t=("float", "uint32_t"),
    int8_t_uint32_t=("int8_t", "uint32_t"),
    uint8_t_uint32_t=("uint8_t", "uint32_t"),
)

for macros, types in [(macros_1, types_1), (macros_2, types_2)]:
  for type_path, (T, IdxT) in types.items():
      for macro_path, macro in macros.items():
          path = f"ann_mg_{macro_path}_{type_path}.cu"
          with open(path, "w") as f:
              f.write(header)
              f.write(macro['include'])
              f.write(namespace_macro)
              f.write(macro["definition"])
              f.write(f"{macro['name']}({T}, {IdxT});\n\n")
              f.write(f"#undef {macro['name']}\n")
              f.write(footer)

          print(f"src/neighbors/ann_mg/{path}")
