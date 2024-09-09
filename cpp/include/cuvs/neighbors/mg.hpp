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

#include <cuvs/neighbors/iface.hpp>

#ifndef NO_NCCL_FORWARD_DECLARATION
class ncclComm_t {};
#endif

#define DEFAULT_SEARCH_BATCH_SIZE 1 << 20

/// \defgroup mg_cpp_index_params ANN MG index build parameters

namespace cuvs::neighbors::mg {
/** Distribution mode */
/// \ingroup mg_cpp_index_params
enum distribution_mode {
  /** Index is replicated on each device, favors throughput */
  REPLICATED,
  /** Index is split on several devices, favors scaling */
  SHARDED
};

/** Merge mode when using a sharded index */
/// \ingroup mg_cpp_index_params
enum sharded_merge_mode {
  /** Search batches are merged on the root rank */
  MERGE_ON_ROOT_RANK,
  /** Search batches are merged in a tree reduction fashion */
  TREE_MERGE
};

template <typename Upstream>
struct index_params : public Upstream {
  /** Distribution mode */
  cuvs::neighbors::mg::distribution_mode mode = SHARDED;
};

template <typename Upstream>
struct search_params : public Upstream {
  /** Sharded search mode */
  cuvs::neighbors::mg::sharded_merge_mode merge_mode = TREE_MERGE;
};

}  // namespace cuvs::neighbors::mg

namespace cuvs::neighbors::mg {
using pool_mr = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;

/// \defgroup mg_cpp_nccl_clique NCCL clique utility

/// \ingroup mg_cpp_nccl_clique
struct nccl_clique {
  /**
   * Instantiates a NCCL clique with all available GPUs
   *
   * @param[in] percent_of_free_memory percentage of device memory to pre-allocate as memory pool
   *
   */
  nccl_clique(int percent_of_free_memory = 80);

  /**
   * Instantiates a NCCL clique
   *
   * Usage example:
   * @code{.cpp}
   * int n_devices;
   * cudaGetDeviceCount(&n_devices);
   * std::vector<int> device_ids(n_devices);
   * std::iota(device_ids.begin(), device_ids.end(), 0);
   * cuvs::neighbors::mg::nccl_clique& clique(device_ids); // first device is the root rank
   * @endcode
   *
   * @param[in] device_ids list of device IDs to be used to initiate the clique
   * @param[in] percent_of_free_memory percentage of device memory to pre-allocate as memory pool
   *
   */
  nccl_clique(const std::vector<int>& device_ids, int percent_of_free_memory = 80);

  void nccl_clique_init();
  const raft::device_resources& set_current_device_to_root_rank() const;
  ~nccl_clique();

  int root_rank_;
  int num_ranks_;
  int percent_of_free_memory_;
  std::vector<int> device_ids_;
  std::vector<ncclComm_t> nccl_comms_;
  std::vector<std::unique_ptr<pool_mr>> per_device_pools_;
  std::vector<raft::device_resources> device_resources_;
};

using namespace raft;

template <typename AnnIndexType, typename T, typename IdxT>
class index {
 public:
  index(distribution_mode mode, int num_ranks_);
  index(const raft::resources& handle,
        const cuvs::neighbors::mg::nccl_clique& clique,
        const std::string& filename);

  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;

  void deserialize_and_distribute(const raft::resources& handle,
                                  const cuvs::neighbors::mg::nccl_clique& clique,
                                  const std::string& filename);

  void deserialize_mg_index(const raft::resources& handle,
                            const cuvs::neighbors::mg::nccl_clique& clique,
                            const std::string& filename);

  void build(const cuvs::neighbors::mg::nccl_clique& clique,
             const cuvs::neighbors::index_params* index_params,
             raft::host_matrix_view<const T, int64_t, row_major> index_dataset);

  void extend(const cuvs::neighbors::mg::nccl_clique& clique,
              raft::host_matrix_view<const T, int64_t, row_major> new_vectors,
              std::optional<raft::host_vector_view<const IdxT, int64_t>> new_indices);

  void search(const cuvs::neighbors::mg::nccl_clique& clique,
              const cuvs::neighbors::search_params* search_params,
              raft::host_matrix_view<const T, int64_t, row_major> queries,
              raft::host_matrix_view<IdxT, int64_t, row_major> neighbors,
              raft::host_matrix_view<float, int64_t, row_major> distances,
              int64_t n_rows_per_batch) const;

  void serialize(raft::resources const& handle,
                 const cuvs::neighbors::mg::nccl_clique& clique,
                 const std::string& filename) const;

 private:
  void sharded_search_with_direct_merge(const cuvs::neighbors::mg::nccl_clique& clique,
                                        const cuvs::neighbors::search_params* search_params,
                                        raft::host_matrix_view<const T, int64_t, row_major> queries,
                                        raft::host_matrix_view<IdxT, int64_t, row_major> neighbors,
                                        raft::host_matrix_view<float, int64_t, row_major> distances,
                                        int64_t n_rows_per_batch,
                                        int64_t n_rows,
                                        int64_t n_cols,
                                        int64_t n_neighbors,
                                        int64_t n_batches) const;

  void sharded_search_with_tree_merge(const cuvs::neighbors::mg::nccl_clique& clique,
                                      const cuvs::neighbors::search_params* search_params,
                                      raft::host_matrix_view<const T, int64_t, row_major> queries,
                                      raft::host_matrix_view<IdxT, int64_t, row_major> neighbors,
                                      raft::host_matrix_view<float, int64_t, row_major> distances,
                                      int64_t n_rows_per_batch,
                                      int64_t n_rows,
                                      int64_t n_cols,
                                      int64_t n_neighbors,
                                      int64_t n_batches) const;

  distribution_mode mode_;
  int num_ranks_;
  std::vector<iface<AnnIndexType, T, IdxT>> ann_interfaces_;
};

/// \defgroup mg_cpp_index_build ANN MG index build

/// \ingroup mg_cpp_index_build
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-Flat MG index
 */
auto build(const raft::resources& handle,
           const cuvs::neighbors::mg::nccl_clique& clique,
           const mg::index_params<ivf_flat::index_params>& index_params,
           raft::host_matrix_view<const float, int64_t, row_major> index_dataset)
  -> index<ivf_flat::index<float, int64_t>, float, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-Flat MG index
 */
auto build(const raft::resources& handle,
           const cuvs::neighbors::mg::nccl_clique& clique,
           const mg::index_params<ivf_flat::index_params>& index_params,
           raft::host_matrix_view<const int8_t, int64_t, row_major> index_dataset)
  -> index<ivf_flat::index<int8_t, int64_t>, int8_t, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-Flat MG index
 */
auto build(const raft::resources& handle,
           const cuvs::neighbors::mg::nccl_clique& clique,
           const mg::index_params<ivf_flat::index_params>& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, row_major> index_dataset)
  -> index<ivf_flat::index<uint8_t, int64_t>, uint8_t, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-PQ MG index
 */
auto build(const raft::resources& handle,
           const cuvs::neighbors::mg::nccl_clique& clique,
           const mg::index_params<ivf_pq::index_params>& index_params,
           raft::host_matrix_view<const float, int64_t, row_major> index_dataset)
  -> index<ivf_pq::index<int64_t>, float, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-PQ MG index
 */
auto build(const raft::resources& handle,
           const cuvs::neighbors::mg::nccl_clique& clique,
           const mg::index_params<ivf_pq::index_params>& index_params,
           raft::host_matrix_view<const int8_t, int64_t, row_major> index_dataset)
  -> index<ivf_pq::index<int64_t>, int8_t, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-PQ MG index
 */
auto build(const raft::resources& handle,
           const cuvs::neighbors::mg::nccl_clique& clique,
           const mg::index_params<ivf_pq::index_params>& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, row_major> index_dataset)
  -> index<ivf_pq::index<int64_t>, uint8_t, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<cagra::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed CAGRA MG index
 */
auto build(const raft::resources& handle,
           const cuvs::neighbors::mg::nccl_clique& clique,
           const mg::index_params<cagra::index_params>& index_params,
           raft::host_matrix_view<const float, int64_t, row_major> index_dataset)
  -> index<cagra::index<float, uint32_t>, float, uint32_t>;

/// \ingroup mg_cpp_index_build
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<cagra::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed CAGRA MG index
 */
auto build(const raft::resources& handle,
           const cuvs::neighbors::mg::nccl_clique& clique,
           const mg::index_params<cagra::index_params>& index_params,
           raft::host_matrix_view<const int8_t, int64_t, row_major> index_dataset)
  -> index<cagra::index<int8_t, uint32_t>, int8_t, uint32_t>;

/// \ingroup mg_cpp_index_build
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<cagra::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed CAGRA MG index
 */
auto build(const raft::resources& handle,
           const cuvs::neighbors::mg::nccl_clique& clique,
           const mg::index_params<cagra::index_params>& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, row_major> index_dataset)
  -> index<cagra::index<uint8_t, uint32_t>, uint8_t, uint32_t>;

/// \defgroup mg_cpp_index_extend ANN MG index extend

/// \ingroup mg_cpp_index_extend
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   cuvs::neighbors::mg::extend(handle, clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            index<ivf_flat::index<float, int64_t>, float, int64_t>& index,
            raft::host_matrix_view<const float, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   cuvs::neighbors::mg::extend(handle, clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            index<ivf_flat::index<int8_t, int64_t>, int8_t, int64_t>& index,
            raft::host_matrix_view<const int8_t, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   cuvs::neighbors::mg::extend(handle, clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            index<ivf_flat::index<uint8_t, int64_t>, uint8_t, int64_t>& index,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   cuvs::neighbors::mg::extend(handle, clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            index<ivf_pq::index<int64_t>, float, int64_t>& index,
            raft::host_matrix_view<const float, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   cuvs::neighbors::mg::extend(handle, clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            index<ivf_pq::index<int64_t>, int8_t, int64_t>& index,
            raft::host_matrix_view<const int8_t, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   cuvs::neighbors::mg::extend(handle, clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            index<ivf_pq::index<int64_t>, uint8_t, int64_t>& index,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<cagra::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   cuvs::neighbors::mg::extend(handle, clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            index<cagra::index<float, uint32_t>, float, uint32_t>& index,
            raft::host_matrix_view<const float, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const uint32_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<cagra::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   cuvs::neighbors::mg::extend(handle, clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            index<cagra::index<int8_t, uint32_t>, int8_t, uint32_t>& index,
            raft::host_matrix_view<const int8_t, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const uint32_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<cagra::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   cuvs::neighbors::mg::extend(handle, clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            index<cagra::index<uint8_t, uint32_t>, uint8_t, uint32_t>& index,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const uint32_t, int64_t>> new_indices);

/// \defgroup mg_cpp_index_search ANN MG index search

/// \ingroup mg_cpp_index_search
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   cuvs::neighbors::mg::search_params<ivf_flat::search_params> search_params; // default search
 * parameters cuvs::neighbors::mg::search(handle, clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            const index<ivf_flat::index<float, int64_t>, float, int64_t>& index,
            const mg::search_params<ivf_flat::search_params>& search_params,
            raft::host_matrix_view<const float, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \ingroup mg_cpp_index_search
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   cuvs::neighbors::mg::search_params<ivf_flat::search_params> search_params; // default search
 * parameters cuvs::neighbors::mg::search(handle, clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            const index<ivf_flat::index<int8_t, int64_t>, int8_t, int64_t>& index,
            const mg::search_params<ivf_flat::search_params>& search_params,
            raft::host_matrix_view<const int8_t, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \ingroup mg_cpp_index_search
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   cuvs::neighbors::mg::search_params<ivf_flat::search_params> search_params; // default search
 * parameters cuvs::neighbors::mg::search(handle, clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            const index<ivf_flat::index<uint8_t, int64_t>, uint8_t, int64_t>& index,
            const mg::search_params<ivf_flat::search_params>& search_params,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \ingroup mg_cpp_index_search
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   cuvs::neighbors::mg::search_params<ivf_pq::search_params> search_params; // default search
 * parameters cuvs::neighbors::mg::search(handle, clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            const index<ivf_pq::index<int64_t>, float, int64_t>& index,
            const mg::search_params<ivf_pq::search_params>& search_params,
            raft::host_matrix_view<const float, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \ingroup mg_cpp_index_search
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   cuvs::neighbors::mg::search_params<ivf_pq::search_params> search_params; // default search
 * parameters cuvs::neighbors::mg::search(handle, clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            const index<ivf_pq::index<int64_t>, int8_t, int64_t>& index,
            const mg::search_params<ivf_pq::search_params>& search_params,
            raft::host_matrix_view<const int8_t, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \ingroup mg_cpp_index_search
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   cuvs::neighbors::mg::search_params<ivf_pq::search_params> search_params; // default search
 * parameters cuvs::neighbors::mg::search(handle, clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            const index<ivf_pq::index<int64_t>, uint8_t, int64_t>& index,
            const mg::search_params<ivf_pq::search_params>& search_params,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \ingroup mg_cpp_index_search
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<cagra::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   cuvs::neighbors::mg::search_params<cagra::search_params> search_params; // default search
 * parameters cuvs::neighbors::mg::search(handle, clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            const index<cagra::index<float, uint32_t>, float, uint32_t>& index,
            const mg::search_params<cagra::search_params>& search_params,
            raft::host_matrix_view<const float, int64_t, row_major> queries,
            raft::host_matrix_view<uint32_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \ingroup mg_cpp_index_search
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<cagra::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   cuvs::neighbors::mg::search_params<cagra::search_params> search_params; // default search
 * parameters cuvs::neighbors::mg::search(handle, clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            const index<cagra::index<int8_t, uint32_t>, int8_t, uint32_t>& index,
            const mg::search_params<cagra::search_params>& search_params,
            raft::host_matrix_view<const int8_t, int64_t, row_major> queries,
            raft::host_matrix_view<uint32_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \ingroup mg_cpp_index_search
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<cagra::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   cuvs::neighbors::mg::search_params<cagra::search_params> search_params; // default search
 * parameters cuvs::neighbors::mg::search(handle, clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::resources& handle,
            const cuvs::neighbors::mg::nccl_clique& clique,
            const index<cagra::index<uint8_t, uint32_t>, uint8_t, uint32_t>& index,
            const mg::search_params<cagra::search_params>& search_params,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> queries,
            raft::host_matrix_view<uint32_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \defgroup mg_cpp_serialize ANN MG index serialization

/// \ingroup mg_cpp_serialize
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   const std::string filename = "mg_index.cuvs";
 *   cuvs::neighbors::mg::serialize(handle, clique, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::resources& handle,
               const cuvs::neighbors::mg::nccl_clique& clique,
               const index<ivf_flat::index<float, int64_t>, float, int64_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   const std::string filename = "mg_index.cuvs";
 *   cuvs::neighbors::mg::serialize(handle, clique, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::resources& handle,
               const cuvs::neighbors::mg::nccl_clique& clique,
               const index<ivf_flat::index<int8_t, int64_t>, int8_t, int64_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   const std::string filename = "mg_index.cuvs";
 *   cuvs::neighbors::mg::serialize(handle, clique, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::resources& handle,
               const cuvs::neighbors::mg::nccl_clique& clique,
               const index<ivf_flat::index<uint8_t, int64_t>, uint8_t, int64_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   const std::string filename = "mg_index.cuvs";
 *   cuvs::neighbors::mg::serialize(handle, clique, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::resources& handle,
               const cuvs::neighbors::mg::nccl_clique& clique,
               const index<ivf_pq::index<int64_t>, float, int64_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   const std::string filename = "mg_index.cuvs";
 *   cuvs::neighbors::mg::serialize(handle, clique, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::resources& handle,
               const cuvs::neighbors::mg::nccl_clique& clique,
               const index<ivf_pq::index<int64_t>, int8_t, int64_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   const std::string filename = "mg_index.cuvs";
 *   cuvs::neighbors::mg::serialize(handle, clique, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::resources& handle,
               const cuvs::neighbors::mg::nccl_clique& clique,
               const index<ivf_pq::index<int64_t>, uint8_t, int64_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<cagra::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   const std::string filename = "mg_index.cuvs";
 *   cuvs::neighbors::mg::serialize(handle, clique, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::resources& handle,
               const cuvs::neighbors::mg::nccl_clique& clique,
               const index<cagra::index<float, uint32_t>, float, uint32_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<cagra::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   const std::string filename = "mg_index.cuvs";
 *   cuvs::neighbors::mg::serialize(handle, clique, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::resources& handle,
               const cuvs::neighbors::mg::nccl_clique& clique,
               const index<cagra::index<int8_t, uint32_t>, int8_t, uint32_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<cagra::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   const std::string filename = "mg_index.cuvs";
 *   cuvs::neighbors::mg::serialize(handle, clique, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::resources& handle,
               const cuvs::neighbors::mg::nccl_clique& clique,
               const index<cagra::index<uint8_t, uint32_t>, uint8_t, uint32_t>& index,
               const std::string& filename);

/// \defgroup mg_cpp_deserialize ANN MG index deserialization

/// \ingroup mg_cpp_deserialize
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   const std::string filename = "mg_index.cuvs";
 *   cuvs::neighbors::mg::serialize(handle, clique, index, filename);
 *   auto new_index = cuvs::neighbors::mg::deserialize_flat<float, int64_t>(handle, clique,
 * filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] filename path to the file to be deserialized
 *
 */
template <typename T, typename IdxT>
auto deserialize_flat(const raft::resources& handle,
                      const cuvs::neighbors::mg::nccl_clique& clique,
                      const std::string& filename) -> index<ivf_flat::index<T, IdxT>, T, IdxT>;

/// \ingroup mg_cpp_deserialize
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   const std::string filename = "mg_index.cuvs";
 *   cuvs::neighbors::mg::serialize(handle, clique, index, filename);
 *   auto new_index = cuvs::neighbors::mg::deserialize_pq<float, int64_t>(handle, clique, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] filename path to the file to be deserialized
 *
 */
template <typename T, typename IdxT>
auto deserialize_pq(const raft::resources& handle,
                    const cuvs::neighbors::mg::nccl_clique& clique,
                    const std::string& filename) -> index<ivf_pq::index<IdxT>, T, IdxT>;

/// \ingroup mg_cpp_deserialize
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   cuvs::neighbors::mg::index_params<cagra::index_params> index_params; // default build
 * parameters auto index = cuvs::neighbors::mg::build(handle, clique, index_params, index_dataset);
 *   const std::string filename = "mg_index.cuvs";
 *   cuvs::neighbors::mg::serialize(handle, clique, index, filename);
 *   auto new_index = cuvs::neighbors::mg::deserialize_cagra<float, uint32_t>(handle, clique,
 * filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] filename path to the file to be deserialized
 *
 */
template <typename T, typename IdxT>
auto deserialize_cagra(const raft::resources& handle,
                       const cuvs::neighbors::mg::nccl_clique& clique,
                       const std::string& filename) -> index<cagra::index<T, IdxT>, T, IdxT>;

/// \defgroup mg_cpp_distribute ANN MG local index distribution

/// \ingroup mg_cpp_distribute
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::ivf_flat::index_params index_params; // default build parameters
 *   auto index = cuvs::neighbors::ivf_flat::build(handle, index_params, index_dataset);
 *   const std::string filename = "local_index.cuvs";
 *   cuvs::neighbors::ivf_flat::serialize(handle, filename, index);
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   auto new_index = cuvs::neighbors::mg::distribute_flat<float, int64_t>(handle, clique,
 * filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] filename path to the file to be deserialized : a local index
 *
 */
template <typename T, typename IdxT>
auto distribute_flat(const raft::resources& handle,
                     const cuvs::neighbors::mg::nccl_clique& clique,
                     const std::string& filename) -> index<ivf_flat::index<T, IdxT>, T, IdxT>;

/// \ingroup mg_cpp_distribute
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::ivf_pq::index_params index_params; // default build parameters
 *   auto index = cuvs::neighbors::ivf_pq::build(handle, index_params, index_dataset);
 *   const std::string filename = "local_index.cuvs";
 *   cuvs::neighbors::ivf_pq::serialize(handle, filename, index);
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   auto new_index = cuvs::neighbors::mg::distribute_pq<float, int64_t>(handle, clique, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] filename path to the file to be deserialized : a local index
 *
 */
template <typename T, typename IdxT>
auto distribute_pq(const raft::resources& handle,
                   const cuvs::neighbors::mg::nccl_clique& clique,
                   const std::string& filename) -> index<ivf_pq::index<IdxT>, T, IdxT>;

/// \ingroup mg_cpp_distribute
/**
 *
 * Usage example:
 * @code{.cpp}
 *   raft::handle_t handle;
 *   cuvs::neighbors::cagra::index_params index_params; // default build parameters
 *   auto index = cuvs::neighbors::cagra::build(handle, index_params, index_dataset);
 *   const std::string filename = "local_index.cuvs";
 *   cuvs::neighbors::cagra::serialize(handle, filename, index);
 *   cuvs::neighbors::mg::nccl_clique& clique; // default NCCL clique
 *   auto new_index = cuvs::neighbors::mg::distribute_cagra<float, uint32_t>(handle, clique,
 * filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] clique configure the NCCL clique
 * @param[in] filename path to the file to be deserialized : a local index
 *
 */
template <typename T, typename IdxT>
auto distribute_cagra(const raft::resources& handle,
                      const cuvs::neighbors::mg::nccl_clique& clique,
                      const std::string& filename) -> index<cagra::index<T, IdxT>, T, IdxT>;

}  // namespace cuvs::neighbors::mg

#else

static_assert(false,
              "FORBIDEN_MG_ALGORITHM_IMPORT\n\n"
              "Please recompile the cuVS library with MG algorithms BUILD_MG_ALGOS=ON.\n");

#endif
