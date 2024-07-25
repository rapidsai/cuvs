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

#pragma once

#ifdef CUVS_BUILD_MG_ALGOS

#include <raft/core/device_resources.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

#ifndef NO_NCCL_FORWARD_DECLARATION
class ncclComm_t {};
#endif

#define DEFAULT_SEARCH_BATCH_SIZE 1 << 20

namespace cuvs::neighbors::mg {
enum distribution_mode { REPLICATED, SHARDED };
}

namespace cuvs::neighbors::ivf_flat {
struct mg_index_params : cuvs::neighbors::ivf_flat::index_params {
  cuvs::neighbors::mg::distribution_mode mode;
};
}  // namespace cuvs::neighbors::ivf_flat

namespace cuvs::neighbors::ivf_pq {
struct mg_index_params : cuvs::neighbors::ivf_pq::index_params {
  cuvs::neighbors::mg::distribution_mode mode;
};
}  // namespace cuvs::neighbors::ivf_pq

namespace cuvs::neighbors::cagra {
struct mg_index_params : cuvs::neighbors::cagra::index_params {
  cuvs::neighbors::mg::distribution_mode mode;
};
}  // namespace cuvs::neighbors::cagra

namespace cuvs::neighbors::mg {
using pool_mr = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;

struct nccl_clique {
  nccl_clique(const std::vector<int>& device_ids, int percent_of_free_memory = 80);
  const raft::device_resources& set_current_device_to_root_rank() const;
  ~nccl_clique();

  int root_rank_;
  int num_ranks_;
  std::vector<int> device_ids_;
  std::vector<ncclComm_t> nccl_comms_;
  std::vector<std::unique_ptr<pool_mr>> per_device_pools_;
  std::vector<raft::device_resources> device_resources_;
};

using namespace raft;

template <typename AnnIndexType, typename T, typename IdxT>
class ann_interface {
 public:
  template <typename Accessor>
  void build(raft::resources const& handle,
             const cuvs::neighbors::index_params* index_params,
             raft::mdspan<const T, matrix_extent<IdxT>, row_major, Accessor> index_dataset);

  template <typename Accessor1, typename Accessor2>
  void extend(
    raft::resources const& handle,
    raft::mdspan<const T, matrix_extent<IdxT>, row_major, Accessor1> new_vectors,
    std::optional<raft::mdspan<const IdxT, vector_extent<IdxT>, layout_c_contiguous, Accessor2>>
      new_indices);

  void search(raft::resources const& handle,
              const cuvs::neighbors::search_params* search_params,
              raft::host_matrix_view<const T, IdxT, row_major> h_queries,
              raft::device_matrix_view<IdxT, IdxT, row_major> d_neighbors,
              raft::device_matrix_view<float, IdxT, row_major> d_distances) const;

  void serialize(raft::resources const& handle, std::ostream& os) const;
  void deserialize(raft::resources const& handle, std::istream& is);
  void deserialize(raft::resources const& handle, const std::string& filename);
  const IdxT size() const;

 private:
  std::optional<AnnIndexType> index_;
};

template <typename AnnIndexType, typename T, typename IdxT>
class ann_mg_index {
 public:
  ann_mg_index(distribution_mode mode, int num_ranks_);
  ann_mg_index(const raft::resources& handle,
               const cuvs::neighbors::mg::nccl_clique& clique,
               const std::string& filename);

  ann_mg_index(const ann_mg_index&)                    = delete;
  ann_mg_index(ann_mg_index&&)                         = default;
  auto operator=(const ann_mg_index&) -> ann_mg_index& = delete;
  auto operator=(ann_mg_index&&) -> ann_mg_index&      = default;

  void deserialize_and_distribute(const raft::resources& handle,
                                  const cuvs::neighbors::mg::nccl_clique& clique,
                                  const std::string& filename);

  void deserialize_mg_index(const raft::resources& handle,
                            const cuvs::neighbors::mg::nccl_clique& clique,
                            const std::string& filename);

  void build(const cuvs::neighbors::mg::nccl_clique& clique,
             const cuvs::neighbors::index_params* index_params,
             raft::host_matrix_view<const T, IdxT, row_major> index_dataset);

  void extend(const cuvs::neighbors::mg::nccl_clique& clique,
              raft::host_matrix_view<const T, IdxT, row_major> new_vectors,
              std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices);

  void search(const cuvs::neighbors::mg::nccl_clique& clique,
              const cuvs::neighbors::search_params* search_params,
              raft::host_matrix_view<const T, IdxT, row_major> queries,
              raft::host_matrix_view<IdxT, IdxT, row_major> neighbors,
              raft::host_matrix_view<float, IdxT, row_major> distances,
              IdxT n_rows_per_batch) const;

  void serialize(raft::resources const& handle,
                 const cuvs::neighbors::mg::nccl_clique& clique,
                 const std::string& filename) const;

 private:
  distribution_mode mode_;
  int num_ranks_;
  std::vector<ann_interface<AnnIndexType, T, IdxT>> ann_interfaces_;
};

#define CUVS_INST_ANN_MG_FLAT(T, IdxT)                                              \
  auto build(const raft::resources& handle,                                         \
             const cuvs::neighbors::mg::nccl_clique& clique,                        \
             const ivf_flat::mg_index_params& index_params,                         \
             raft::host_matrix_view<const T, IdxT, row_major> index_dataset)        \
    ->ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>;                              \
                                                                                    \
  void extend(const raft::resources& handle,                                        \
              const cuvs::neighbors::mg::nccl_clique& clique,                       \
              ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,               \
              raft::host_matrix_view<const T, IdxT, row_major> new_vectors,         \
              std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices); \
                                                                                    \
  void search(const raft::resources& handle,                                        \
              const cuvs::neighbors::mg::nccl_clique& clique,                       \
              const ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,         \
              const ivf_flat::search_params& search_params,                         \
              raft::host_matrix_view<const T, IdxT, row_major> queries,             \
              raft::host_matrix_view<IdxT, IdxT, row_major> neighbors,              \
              raft::host_matrix_view<float, IdxT, row_major> distances,             \
              uint64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);               \
                                                                                    \
  void serialize(const raft::resources& handle,                                     \
                 const cuvs::neighbors::mg::nccl_clique& clique,                    \
                 const ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,      \
                 const std::string& filename);

CUVS_INST_ANN_MG_FLAT(float, int64_t);
CUVS_INST_ANN_MG_FLAT(int8_t, int64_t);
CUVS_INST_ANN_MG_FLAT(uint8_t, int64_t);

#undef CUVS_INST_ANN_MG_FLAT

template <typename T, typename IdxT>
auto deserialize_flat(const raft::resources& handle,
                      const cuvs::neighbors::mg::nccl_clique& clique,
                      const std::string& filename)
  -> ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>;

template <typename T, typename IdxT>
auto distribute_flat(const raft::resources& handle,
                     const cuvs::neighbors::mg::nccl_clique& clique,
                     const std::string& filename)
  -> ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>;

#define CUVS_INST_ANN_MG_PQ(T, IdxT)                                                \
  auto build(const raft::resources& handle,                                         \
             const cuvs::neighbors::mg::nccl_clique& clique,                        \
             const ivf_pq::mg_index_params& index_params,                           \
             raft::host_matrix_view<const T, IdxT, row_major> index_dataset)        \
    ->ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>;                                   \
                                                                                    \
  void extend(const raft::resources& handle,                                        \
              const cuvs::neighbors::mg::nccl_clique& clique,                       \
              ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>& index,                    \
              raft::host_matrix_view<const T, IdxT, row_major> new_vectors,         \
              std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices); \
                                                                                    \
  void search(const raft::resources& handle,                                        \
              const cuvs::neighbors::mg::nccl_clique& clique,                       \
              const ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>& index,              \
              const ivf_pq::search_params& search_params,                           \
              raft::host_matrix_view<const T, IdxT, row_major> queries,             \
              raft::host_matrix_view<IdxT, IdxT, row_major> neighbors,              \
              raft::host_matrix_view<float, IdxT, row_major> distances,             \
              uint64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);               \
                                                                                    \
  void serialize(const raft::resources& handle,                                     \
                 const cuvs::neighbors::mg::nccl_clique& clique,                    \
                 const ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>& index,           \
                 const std::string& filename);

CUVS_INST_ANN_MG_PQ(float, int64_t);
CUVS_INST_ANN_MG_PQ(int8_t, int64_t);
CUVS_INST_ANN_MG_PQ(uint8_t, int64_t);

#undef CUVS_INST_ANN_MG_PQ

template <typename T, typename IdxT>
auto deserialize_pq(const raft::resources& handle,
                    const cuvs::neighbors::mg::nccl_clique& clique,
                    const std::string& filename) -> ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>;

template <typename T, typename IdxT>
auto distribute_pq(const raft::resources& handle,
                   const cuvs::neighbors::mg::nccl_clique& clique,
                   const std::string& filename) -> ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>;

#define CUVS_INST_ANN_MG_CAGRA(T, IdxT)                                             \
  auto build(const raft::resources& handle,                                         \
             const cuvs::neighbors::mg::nccl_clique& clique,                        \
             const cagra::mg_index_params& index_params,                            \
             raft::host_matrix_view<const T, IdxT, row_major> index_dataset)        \
    ->ann_mg_index<cagra::index<T, IdxT>, T, IdxT>;                                 \
                                                                                    \
  void extend(const raft::resources& handle,                                        \
              const cuvs::neighbors::mg::nccl_clique& clique,                       \
              ann_mg_index<cagra::index<T, IdxT>, T, IdxT>& index,                  \
              raft::host_matrix_view<const T, IdxT, row_major> new_vectors,         \
              std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices); \
                                                                                    \
  void search(const raft::resources& handle,                                        \
              const cuvs::neighbors::mg::nccl_clique& clique,                       \
              const ann_mg_index<cagra::index<T, IdxT>, T, IdxT>& index,            \
              const cagra::search_params& search_params,                            \
              raft::host_matrix_view<const T, IdxT, row_major> queries,             \
              raft::host_matrix_view<IdxT, IdxT, row_major> neighbors,              \
              raft::host_matrix_view<float, IdxT, row_major> distances,             \
              uint64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);               \
                                                                                    \
  void serialize(const raft::resources& handle,                                     \
                 const cuvs::neighbors::mg::nccl_clique& clique,                    \
                 const ann_mg_index<cagra::index<T, IdxT>, T, IdxT>& index,         \
                 const std::string& filename);

CUVS_INST_ANN_MG_CAGRA(float, uint32_t);
CUVS_INST_ANN_MG_CAGRA(int8_t, uint32_t);
CUVS_INST_ANN_MG_CAGRA(uint8_t, uint32_t);

#undef CUVS_INST_ANN_MG_CAGRA

template <typename T, typename IdxT>
auto deserialize_cagra(const raft::resources& handle,
                       const cuvs::neighbors::mg::nccl_clique& clique,
                       const std::string& filename) -> ann_mg_index<cagra::index<T, IdxT>, T, IdxT>;

template <typename T, typename IdxT>
auto distribute_cagra(const raft::resources& handle,
                      const cuvs::neighbors::mg::nccl_clique& clique,
                      const std::string& filename) -> ann_mg_index<cagra::index<T, IdxT>, T, IdxT>;

}  // namespace cuvs::neighbors::mg

#else

static_assert(false,
              "FORBIDEN_MG_ALGORITHM_IMPORT\n\n"
              "Please recompile the cuVS library with MG algorithms BUILD_MG_ALGOS=ON.\n");

#endif
