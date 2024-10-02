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

#include <atomic>
#include <memory>

#include <raft/core/device_resources.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

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

/// \defgroup mg_cpp_search_params ANN MG search parameters

/** Search mode when using a replicated index */
/// \ingroup mg_cpp_search_params
enum replicated_search_mode {
  /** Search queries are splited to maintain equal load on GPUs */
  LOAD_BALANCER,
  /** Each search query is processed by a single GPU in a round-robin fashion */
  ROUND_ROBIN
};

/** Merge mode when using a sharded index */
/// \ingroup mg_cpp_search_params
enum sharded_merge_mode {
  /** Search batches are merged on the root rank */
  MERGE_ON_ROOT_RANK,
  /** Search batches are merged in a tree reduction fashion */
  TREE_MERGE
};

/** Build parameters */
/// \ingroup mg_cpp_index_params
template <typename Upstream>
struct index_params : public Upstream {
  /** Distribution mode */
  cuvs::neighbors::mg::distribution_mode mode = SHARDED;
};

/** Search parameters */
/// \ingroup mg_cpp_search_params
template <typename Upstream>
struct search_params : public Upstream {
  /** Replicated search mode */
  cuvs::neighbors::mg::replicated_search_mode search_mode = LOAD_BALANCER;
  /** Sharded merge mode */
  cuvs::neighbors::mg::sharded_merge_mode merge_mode = TREE_MERGE;
};

}  // namespace cuvs::neighbors::mg

namespace cuvs::neighbors::mg {

using namespace raft;

template <typename AnnIndexType, typename T, typename IdxT>
struct index {
  index(distribution_mode mode, int num_ranks_);
  index(const raft::device_resources& handle, const std::string& filename);

  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;

  distribution_mode mode_;
  int num_ranks_;
  std::vector<iface<AnnIndexType, T, IdxT>> ann_interfaces_;

  // for load balancing mechanism
  std::shared_ptr<std::atomic<int64_t>> round_robin_counter_;
};

/// \defgroup mg_cpp_index_build ANN MG index build

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-Flat MG index
 */
auto build(const raft::device_resources& handle,
           const mg::index_params<ivf_flat::index_params>& index_params,
           raft::host_matrix_view<const float, int64_t, row_major> index_dataset)
  -> index<ivf_flat::index<float, int64_t>, float, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-Flat MG index
 */
auto build(const raft::device_resources& handle,
           const mg::index_params<ivf_flat::index_params>& index_params,
           raft::host_matrix_view<const int8_t, int64_t, row_major> index_dataset)
  -> index<ivf_flat::index<int8_t, int64_t>, int8_t, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-Flat MG index
 */
auto build(const raft::device_resources& handle,
           const mg::index_params<ivf_flat::index_params>& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, row_major> index_dataset)
  -> index<ivf_flat::index<uint8_t, int64_t>, uint8_t, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-PQ MG index
 */
auto build(const raft::device_resources& handle,
           const mg::index_params<ivf_pq::index_params>& index_params,
           raft::host_matrix_view<const float, int64_t, row_major> index_dataset)
  -> index<ivf_pq::index<int64_t>, float, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-PQ MG index
 */
auto build(const raft::device_resources& handle,
           const mg::index_params<ivf_pq::index_params>& index_params,
           raft::host_matrix_view<const half, int64_t, row_major> index_dataset)
  -> index<ivf_pq::index<int64_t>, half, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-PQ MG index
 */
auto build(const raft::device_resources& handle,
           const mg::index_params<ivf_pq::index_params>& index_params,
           raft::host_matrix_view<const int8_t, int64_t, row_major> index_dataset)
  -> index<ivf_pq::index<int64_t>, int8_t, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-PQ MG index
 */
auto build(const raft::device_resources& handle,
           const mg::index_params<ivf_pq::index_params>& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, row_major> index_dataset)
  -> index<ivf_pq::index<int64_t>, uint8_t, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed CAGRA MG index
 */
auto build(const raft::device_resources& handle,
           const mg::index_params<cagra::index_params>& index_params,
           raft::host_matrix_view<const float, int64_t, row_major> index_dataset)
  -> index<cagra::index<float, uint32_t>, float, uint32_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed CAGRA MG index
 */
auto build(const raft::device_resources& handle,
           const mg::index_params<cagra::index_params>& index_params,
           raft::host_matrix_view<const half, int64_t, row_major> index_dataset)
  -> index<cagra::index<half, uint32_t>, half, uint32_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed CAGRA MG index
 */
auto build(const raft::device_resources& handle,
           const mg::index_params<cagra::index_params>& index_params,
           raft::host_matrix_view<const int8_t, int64_t, row_major> index_dataset)
  -> index<cagra::index<int8_t, uint32_t>, int8_t, uint32_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed CAGRA MG index
 */
auto build(const raft::device_resources& handle,
           const mg::index_params<cagra::index_params>& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, row_major> index_dataset)
  -> index<cagra::index<uint8_t, uint32_t>, uint8_t, uint32_t>;

/// \defgroup mg_cpp_index_extend ANN MG index extend

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::extend(handle, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::device_resources& handle,
            index<ivf_flat::index<float, int64_t>, float, int64_t>& index,
            raft::host_matrix_view<const float, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::extend(handle, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::device_resources& handle,
            index<ivf_flat::index<int8_t, int64_t>, int8_t, int64_t>& index,
            raft::host_matrix_view<const int8_t, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::extend(handle, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::device_resources& handle,
            index<ivf_flat::index<uint8_t, int64_t>, uint8_t, int64_t>& index,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::extend(handle, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::device_resources& handle,
            index<ivf_pq::index<int64_t>, float, int64_t>& index,
            raft::host_matrix_view<const float, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::extend(handle, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::device_resources& handle,
            index<ivf_pq::index<int64_t>, half, int64_t>& index,
            raft::host_matrix_view<const half, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::extend(handle, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::device_resources& handle,
            index<ivf_pq::index<int64_t>, int8_t, int64_t>& index,
            raft::host_matrix_view<const int8_t, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::extend(handle, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::device_resources& handle,
            index<ivf_pq::index<int64_t>, uint8_t, int64_t>& index,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::extend(handle, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::device_resources& handle,
            index<cagra::index<float, uint32_t>, float, uint32_t>& index,
            raft::host_matrix_view<const float, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const uint32_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::extend(handle, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::device_resources& handle,
            index<cagra::index<half, uint32_t>, half, uint32_t>& index,
            raft::host_matrix_view<const half, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const uint32_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::extend(handle, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::device_resources& handle,
            index<cagra::index<int8_t, uint32_t>, int8_t, uint32_t>& index,
            raft::host_matrix_view<const int8_t, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const uint32_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::extend(handle, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::device_resources& handle,
            index<cagra::index<uint8_t, uint32_t>, uint8_t, uint32_t>& index,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const uint32_t, int64_t>> new_indices);

/// \defgroup mg_cpp_index_search ANN MG index search

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::search_params<ivf_flat::search_params> search_params;
 * cuvs::neighbors::mg::search(handle, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::device_resources& handle,
            const index<ivf_flat::index<float, int64_t>, float, int64_t>& index,
            const mg::search_params<ivf_flat::search_params>& search_params,
            raft::host_matrix_view<const float, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::search_params<ivf_flat::search_params> search_params;
 * cuvs::neighbors::mg::search(handle, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::device_resources& handle,
            const index<ivf_flat::index<int8_t, int64_t>, int8_t, int64_t>& index,
            const mg::search_params<ivf_flat::search_params>& search_params,
            raft::host_matrix_view<const int8_t, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::search_params<ivf_flat::search_params> search_params;
 * cuvs::neighbors::mg::search(handle, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::device_resources& handle,
            const index<ivf_flat::index<uint8_t, int64_t>, uint8_t, int64_t>& index,
            const mg::search_params<ivf_flat::search_params>& search_params,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::search_params<ivf_pq::search_params> search_params;
 * cuvs::neighbors::mg::search(handle, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::device_resources& handle,
            const index<ivf_pq::index<int64_t>, float, int64_t>& index,
            const mg::search_params<ivf_pq::search_params>& search_params,
            raft::host_matrix_view<const float, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::search_params<ivf_pq::search_params> search_params;
 * cuvs::neighbors::mg::search(handle, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::device_resources& handle,
            const index<ivf_pq::index<int64_t>, half, int64_t>& index,
            const mg::search_params<ivf_pq::search_params>& search_params,
            raft::host_matrix_view<const half, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::search_params<ivf_pq::search_params> search_params;
 * cuvs::neighbors::mg::search(handle, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::device_resources& handle,
            const index<ivf_pq::index<int64_t>, int8_t, int64_t>& index,
            const mg::search_params<ivf_pq::search_params>& search_params,
            raft::host_matrix_view<const int8_t, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::search_params<ivf_pq::search_params> search_params;
 * cuvs::neighbors::mg::search(handle, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::device_resources& handle,
            const index<ivf_pq::index<int64_t>, uint8_t, int64_t>& index,
            const mg::search_params<ivf_pq::search_params>& search_params,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::search_params<cagra::search_params> search_params;
 * cuvs::neighbors::mg::search(handle, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::device_resources& handle,
            const index<cagra::index<float, uint32_t>, float, uint32_t>& index,
            const mg::search_params<cagra::search_params>& search_params,
            raft::host_matrix_view<const float, int64_t, row_major> queries,
            raft::host_matrix_view<uint32_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::search_params<cagra::search_params> search_params;
 * cuvs::neighbors::mg::search(handle, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::device_resources& handle,
            const index<cagra::index<half, uint32_t>, half, uint32_t>& index,
            const mg::search_params<cagra::search_params>& search_params,
            raft::host_matrix_view<const half, int64_t, row_major> queries,
            raft::host_matrix_view<uint32_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::search_params<cagra::search_params> search_params;
 * cuvs::neighbors::mg::search(handle, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::device_resources& handle,
            const index<cagra::index<int8_t, uint32_t>, int8_t, uint32_t>& index,
            const mg::search_params<cagra::search_params>& search_params,
            raft::host_matrix_view<const int8_t, int64_t, row_major> queries,
            raft::host_matrix_view<uint32_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * cuvs::neighbors::mg::search_params<cagra::search_params> search_params;
 * cuvs::neighbors::mg::search(handle, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 * @param[in] n_rows_per_batch (optional) search batch size
 *
 */
void search(const raft::device_resources& handle,
            const index<cagra::index<uint8_t, uint32_t>, uint8_t, uint32_t>& index,
            const mg::search_params<cagra::search_params>& search_params,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> queries,
            raft::host_matrix_view<uint32_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances,
            int64_t n_rows_per_batch = DEFAULT_SEARCH_BATCH_SIZE);

/// \defgroup mg_cpp_serialize ANN MG index serialization

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::mg::serialize(handle, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::device_resources& handle,
               const index<ivf_flat::index<float, int64_t>, float, int64_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::mg::serialize(handle, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::device_resources& handle,
               const index<ivf_flat::index<int8_t, int64_t>, int8_t, int64_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::mg::serialize(handle, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::device_resources& handle,
               const index<ivf_flat::index<uint8_t, int64_t>, uint8_t, int64_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::mg::serialize(handle, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::device_resources& handle,
               const index<ivf_pq::index<int64_t>, float, int64_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::mg::serialize(handle, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::device_resources& handle,
               const index<ivf_pq::index<int64_t>, half, int64_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::mg::serialize(handle, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::device_resources& handle,
               const index<ivf_pq::index<int64_t>, int8_t, int64_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::mg::serialize(handle, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::device_resources& handle,
               const index<ivf_pq::index<int64_t>, uint8_t, int64_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::mg::serialize(handle, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::device_resources& handle,
               const index<cagra::index<float, uint32_t>, float, uint32_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::mg::serialize(handle, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::device_resources& handle,
               const index<cagra::index<half, uint32_t>, half, uint32_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::mg::serialize(handle, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::device_resources& handle,
               const index<cagra::index<int8_t, uint32_t>, int8_t, uint32_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::mg::serialize(handle, index, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::device_resources& handle,
               const index<cagra::index<uint8_t, uint32_t>, uint8_t, uint32_t>& index,
               const std::string& filename);

/// \defgroup mg_cpp_deserialize ANN MG index deserialization

/// \ingroup mg_cpp_deserialize
/**
 * @brief Deserializes an IVF-Flat multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_flat::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::mg::serialize(handle, index, filename);
 * auto new_index = cuvs::neighbors::mg::deserialize_flat<float, int64_t>(handle, filename);
 *
 * @endcode
 *
 * @param[in] handle
 * @param[in] filename path to the file to be deserialized
 *
 */
template <typename T, typename IdxT>
auto deserialize_flat(const raft::device_resources& handle, const std::string& filename)
  -> index<ivf_flat::index<T, IdxT>, T, IdxT>;

/// \ingroup mg_cpp_deserialize
/**
 * @brief Deserializes an IVF-PQ multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::mg::serialize(handle, index, filename);
 * auto new_index = cuvs::neighbors::mg::deserialize_pq<float, int64_t>(handle, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] filename path to the file to be deserialized
 *
 */
template <typename T, typename IdxT>
auto deserialize_pq(const raft::device_resources& handle, const std::string& filename)
  -> index<ivf_pq::index<IdxT>, T, IdxT>;

/// \ingroup mg_cpp_deserialize
/**
 * @brief Deserializes a CAGRA multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::mg::index_params<cagra::index_params> index_params;
 * auto index = cuvs::neighbors::mg::build(handle, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::mg::serialize(handle, index, filename);
 * auto new_index = cuvs::neighbors::mg::deserialize_cagra<float, uint32_t>(handle, filename);
 *
 * @endcode
 *
 * @param[in] handle
 * @param[in] filename path to the file to be deserialized
 *
 */
template <typename T, typename IdxT>
auto deserialize_cagra(const raft::device_resources& handle, const std::string& filename)
  -> index<cagra::index<T, IdxT>, T, IdxT>;

/// \defgroup mg_cpp_distribute ANN MG local index distribution

/// \ingroup mg_cpp_distribute
/**
 * @brief Replicates a locally built and serialized IVF-Flat index to all GPUs to form a distributed
 * multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::ivf_flat::index_params index_params;
 * auto index = cuvs::neighbors::ivf_flat::build(handle, index_params, index_dataset);
 * const std::string filename = "local_index.cuvs";
 * cuvs::neighbors::ivf_flat::serialize(handle, filename, index);
 * auto new_index = cuvs::neighbors::mg::distribute_flat<float, int64_t>(handle, filename);
 *
 * @endcode
 *
 * @param[in] handle
 * @param[in] filename path to the file to be deserialized : a local index
 *
 */
template <typename T, typename IdxT>
auto distribute_flat(const raft::device_resources& handle, const std::string& filename)
  -> index<ivf_flat::index<T, IdxT>, T, IdxT>;

/// \ingroup mg_cpp_distribute
/**
 * @brief Replicates a locally built and serialized IVF-PQ index to all GPUs to form a distributed
 * multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::ivf_pq::index_params index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(handle, index_params, index_dataset);
 * const std::string filename = "local_index.cuvs";
 * cuvs::neighbors::ivf_pq::serialize(handle, filename, index);
 * auto new_index = cuvs::neighbors::mg::distribute_pq<float, int64_t>(handle, filename);
 * @endcode
 *
 * @param[in] handle
 * @param[in] filename path to the file to be deserialized : a local index
 *
 */
template <typename T, typename IdxT>
auto distribute_pq(const raft::device_resources& handle, const std::string& filename)
  -> index<ivf_pq::index<IdxT>, T, IdxT>;

/// \ingroup mg_cpp_distribute
/**
 * @brief Replicates a locally built and serialized CAGRA index to all GPUs to form a distributed
 * multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::neighbors::cagra::index_params index_params;
 * auto index = cuvs::neighbors::cagra::build(handle, index_params, index_dataset);
 * const std::string filename = "local_index.cuvs";
 * cuvs::neighbors::cagra::serialize(handle, filename, index);
 * auto new_index = cuvs::neighbors::mg::distribute_cagra<float, uint32_t>(handle, filename);
 *
 * @endcode
 *
 * @param[in] handle
 * @param[in] filename path to the file to be deserialized : a local index
 *
 */
template <typename T, typename IdxT>
auto distribute_cagra(const raft::device_resources& handle, const std::string& filename)
  -> index<cagra::index<T, IdxT>, T, IdxT>;

}  // namespace cuvs::neighbors::mg

#else

static_assert(false,
              "FORBIDEN_MG_ALGORITHM_IMPORT\n\n"
              "Please recompile the cuVS library with MG algorithms BUILD_MG_ALGOS=ON.\n");

#endif
