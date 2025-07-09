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

#pragma once

#include "common.hpp"
#include <cstdint>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>

namespace cuvs::neighbors::binary_ivf {
/**
 * @defgroup binary_ivf_cpp_index_params Binary-IVF index build parameters
 * @{
 */

/** Size of the interleaved group (see `index::data` description). */
constexpr static uint32_t kIndexGroupSize = 32;

using index_params = cuvs::neighbors::ivf_flat::index_params;
/**
 * @}
 */

/**
 * @defgroup binary_ivf_cpp_search_params Binary-IVF index search parameters
 * @{
 */
using search_params = cuvs::neighbors::ivf_flat::search_params;

static_assert(std::is_aggregate_v<index_params>);
static_assert(std::is_aggregate_v<search_params>);

template <typename SizeT, typename ValueT, typename IdxT>
struct list_spec {
  using value_type   = ValueT;
  using list_extents = raft::matrix_extent<SizeT>;
  using index_type   = IdxT;

  SizeT align_max;
  SizeT align_min;
  uint32_t dim;

  constexpr list_spec(uint32_t dim, bool conservative_memory_allocation)
    : dim(dim),
      align_min(kIndexGroupSize),
      align_max(conservative_memory_allocation ? kIndexGroupSize : 1024)
  {
  }

  // Allow casting between different size-types (for safer size and offset calculations)
  template <typename OtherSizeT>
  constexpr explicit list_spec(const list_spec<OtherSizeT, ValueT, IdxT>& other_spec)
    : dim{other_spec.dim}, align_min{other_spec.align_min}, align_max{other_spec.align_max}
  {
  }

  /** Determine the extents of an array enough to hold a given amount of data. */
  constexpr auto make_list_extents(SizeT n_rows) const -> list_extents
  {
    return raft::make_extents<SizeT>(n_rows, dim);
  }
};

template <typename ValueT, typename IdxT, typename SizeT = uint32_t>
using list_data = ivf::list<list_spec, SizeT, ValueT, IdxT>;

/**
 * @}
 */

/**
 * @defgroup binary_ivf_cpp_index Binary-IVF index
 * @{
 */
/**
 * @brief Binary-IVF index.
 *
 * @tparam IdxT type of the indices in the source dataset
 *
 */
template <typename IdxT>
struct index : cuvs::neighbors::index {
  using index_params_type  = binary_ivf::index_params;
  using search_params_type = binary_ivf::search_params;
  using index_type         = IdxT;
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

 public:
  index(const index&)            = delete;
  index(index&&)                 = default;
  index& operator=(const index&) = delete;
  index& operator=(index&&)      = default;
  ~index()                       = default;

  /**
   * @brief Construct an empty index.
   *
   * Constructs an empty index. This index will either need to be trained with `build`
   * or loaded from a saved copy with `deserialize`
   */
  index(raft::resources const& res);

  /** Construct an empty index. It needs to be trained and then populated. */
  index(raft::resources const& res, const index_params& params, uint32_t dim);
  /** Construct an empty index. It needs to be trained and then populated. */
  index(raft::resources const& res,
        uint32_t n_lists,
        bool adaptive_centers,
        bool conservative_memory_allocation,
        uint32_t dim);

  /**
   * Vectorized load/store size in elements, determines the size of interleaved data chunks.
   */
  uint32_t veclen() const noexcept;

  /** Distance metric used for clustering. */
  cuvs::distance::DistanceType metric() const noexcept;

  /** Whether `centers()` change upon extending the index (binary_ivf::extend). */
  bool adaptive_centers() const noexcept;

  /**
   * Inverted list data [size, dim].
   *
   * The data consists of the dataset rows, grouped by their labels (into clusters/lists).
   * Within each list (cluster), the data is grouped into blocks of `kIndexGroupSize` interleaved
   * vectors. Note, the total index length is slightly larger than the source dataset length,
   * because each cluster is padded by `kIndexGroupSize` elements.
   *
   * Interleaving pattern:
   * within groups of `kIndexGroupSize` rows, the data is interleaved with the block size equal to
   * `veclen * sizeof(T)`. That is, a chunk of `veclen` consecutive components of one row is
   * followed by a chunk of the same size of the next row, and so on.
   *
   * __Example__: veclen = 2, dim = 6, kIndexGroupSize = 32, list_size = 31
   *
   *     x[ 0, 0], x[ 0, 1], x[ 1, 0], x[ 1, 1], ... x[14, 0], x[14, 1], x[15, 0], x[15, 1],
   *     x[16, 0], x[16, 1], x[17, 0], x[17, 1], ... x[30, 0], x[30, 1],    -    ,    -    ,
   *     x[ 0, 2], x[ 0, 3], x[ 1, 2], x[ 1, 3], ... x[14, 2], x[14, 3], x[15, 2], x[15, 3],
   *     x[16, 2], x[16, 3], x[17, 2], x[17, 3], ... x[30, 2], x[30, 3],    -    ,    -    ,
   *     x[ 0, 4], x[ 0, 5], x[ 1, 4], x[ 1, 5], ... x[14, 4], x[14, 5], x[15, 4], x[15, 5],
   *     x[16, 4], x[16, 5], x[17, 4], x[17, 5], ... x[30, 4], x[30, 5],    -    ,    -    ,
   *
   */
  /** Sizes of the lists (clusters) [n_lists]
   * NB: This may differ from the actual list size if the shared lists have been extended by another
   * index
   */
  raft::device_vector_view<uint32_t, uint32_t> list_sizes() noexcept;
  raft::device_vector_view<const uint32_t, uint32_t> list_sizes() const noexcept;

  /** k-means cluster centers corresponding to the lists [n_lists, dim] */
  raft::device_matrix_view<float, uint32_t, raft::row_major> centers() noexcept;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers() const noexcept;

  /**
   * (Optional) Precomputed norms of the `centers` w.r.t. the chosen distance metric [n_lists].
   *
   * NB: this may be empty if the index is empty or if the metric does not require the center norms
   * calculation.
   */
  std::optional<raft::device_vector_view<float, uint32_t>> center_norms() noexcept;
  std::optional<raft::device_vector_view<const float, uint32_t>> center_norms() const noexcept;

  /**
   * Accumulated list sizes, sorted in descending order [n_lists + 1].
   * The last value contains the total length of the index.
   * The value at index zero is always zero.
   *
   * That is, the content of this span is as if the `list_sizes` was sorted and then accumulated.
   *
   * This span is used during search to estimate the maximum size of the workspace.
   */
  auto accum_sorted_sizes() noexcept -> raft::host_vector_view<IdxT, uint32_t>;
  [[nodiscard]] auto accum_sorted_sizes() const noexcept
    -> raft::host_vector_view<const IdxT, uint32_t>;

  /** Total length of the index. */
  IdxT size() const noexcept;

  /** Dimensionality of the data. */
  uint32_t dim() const noexcept;

  /** Number of clusters/inverted lists. */
  uint32_t n_lists() const noexcept;
  raft::device_vector_view<uint8_t*, uint32_t> data_ptrs() noexcept;
  raft::device_vector_view<uint8_t* const, uint32_t> data_ptrs() const noexcept;

  /** Pointers to the inverted lists (clusters) indices  [n_lists]. */
  raft::device_vector_view<IdxT*, uint32_t> inds_ptrs() noexcept;
  raft::device_vector_view<IdxT* const, uint32_t> inds_ptrs() const noexcept;

  /**
   * Whether to use convervative memory allocation when extending the list (cluster) data
   * (see index_params.conservative_memory_allocation).
   */
  bool conservative_memory_allocation() const noexcept;

  void allocate_center_norms(raft::resources const& res);

  /** Lists' data and indices. */
  std::vector<std::shared_ptr<list_data<uint8_t, IdxT>>>& lists() noexcept;
  const std::vector<std::shared_ptr<list_data<uint8_t, IdxT>>>& lists() const noexcept;

  void check_consistency();

 private:
  /**
   * TODO: in theory, we can lift this to the template parameter and keep it at hardware maximum
   * possible value by padding the `dim` of the data https://github.com/rapidsai/raft/issues/711
   */
  uint32_t veclen_;
  cuvs::distance::DistanceType metric_;
  bool adaptive_centers_;
  bool conservative_memory_allocation_;
  std::vector<std::shared_ptr<list_data<uint8_t, IdxT>>> lists_;
  raft::device_vector<uint32_t, uint32_t> list_sizes_;
  raft::device_matrix<float, uint32_t, raft::row_major> centers_;
  std::optional<raft::device_vector<float, uint32_t>> center_norms_;

  // Computed members
  raft::device_vector<uint8_t*, uint32_t> data_ptrs_;
  raft::device_vector<IdxT*, uint32_t> inds_ptrs_;
  raft::host_vector<IdxT, uint32_t> accum_sorted_sizes_;

  static auto calculate_veclen(uint32_t dim) -> uint32_t
  {
    // TODO: consider padding the dimensions and fixing veclen to its maximum possible value as a
    // template parameter (https://github.com/rapidsai/raft/issues/711)

    // NOTE: keep this consistent with the select_interleaved_scan_kernel logic
    // in detail/binary_ivf_interleaved_scan-inl.cuh.
    uint32_t veclen = std::max<uint32_t>(1, 16);
    if (dim % veclen != 0) { veclen = 1; }
    return veclen;
  }
};
/**
 * @}
 */

/**
 * @defgroup binary_ivf_cpp_index_build IVF-Flat index build
 * @{
 */
/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   binary_ivf::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = binary_ivf::build(handle, dataset, index_params);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-flat index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::binary_ivf::index_params& index_params,
           raft::device_matrix_view<const uint8_t*, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::binary_ivf::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   binary_ivf::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   binary_ivf::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   binary_ivf::build(handle, dataset, index_params, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::device_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to binary_ivf::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::binary_ivf::index_params& index_params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::binary_ivf::index<int64_t>& idx);

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   binary_ivf::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   auto index = binary_ivf::build(handle, dataset, index_params);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-flat index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::binary_ivf::index_params& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::binary_ivf::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   binary_ivf::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   binary_ivf::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   binary_ivf::build(handle, dataset, index_params, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to binary_ivf::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::binary_ivf::index_params& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::binary_ivf::index<int64_t>& idx);
/**
 * @}
 */

/**
 * @defgroup binary_ivf_cpp_index_extend IVF-Flat index extend
 * @{
 */

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are adjusted to match the newly labeled data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   binary_ivf::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = binary_ivf::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = binary_ivf::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors raft::device_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::device_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] idx original index
 *
 * @return the constructed extended ivf-flat index
 */
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::binary_ivf::index<float, int64_t>& idx)
  -> cuvs::neighbors::binary_ivf::index<float, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   binary_ivf::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = binary_ivf::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   binary_ivf::extend(handle, dataset, no_opt, &index_empty);
 * @endcode
 *
 *
 * @param[in] handle
 * @param[in] new_vectors raft::device_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::device_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to index, to be overwritten in-place
 */
void extend(raft::resources const& handle,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::binary_ivf::index<float, int64_t>* idx);

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are adjusted to match the newly labeled data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   binary_ivf::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = binary_ivf::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = binary_ivf::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors raft::device_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::device_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] idx original index
 *
 * @return the constructed extended ivf-flat index
 */
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::binary_ivf::index<half, int64_t>& idx)
  -> cuvs::neighbors::binary_ivf::index<half, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   binary_ivf::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = binary_ivf::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   binary_ivf::extend(handle, dataset, no_opt, &index_empty);
 * @endcode
 *
 *
 * @param[in] handle
 * @param[in] new_vectors raft::device_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::device_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to index, to be overwritten in-place
 */
void extend(raft::resources const& handle,
            raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::binary_ivf::index<half, int64_t>* idx);

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are adjusted to match the newly labeled data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   binary_ivf::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = binary_ivf::build(handle, dataset, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = binary_ivf::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors raft::device_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::device_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] idx original index
 *
 * @return the constructed extended ivf-flat index
 */
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::binary_ivf::index<int8_t, int64_t>& idx)
  -> cuvs::neighbors::binary_ivf::index<int8_t, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   binary_ivf::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = binary_ivf::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   binary_ivf::extend(handle, dataset, no_opt, &index_empty);
 * @endcode
 *
 *
 * @param[in] handle
 * @param[in] new_vectors raft::device_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::device_vector_view to a vector of indices [n_rows].
 *
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to index, to be overwritten in-place
 */
void extend(raft::resources const& handle,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::binary_ivf::index<int8_t, int64_t>* idx);

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are adjusted to match the newly labeled data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   binary_ivf::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = binary_ivf::build(handle, dataset, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = binary_ivf::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors raft::device_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::device_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] idx original index
 *
 * @return the constructed extended ivf-flat index
 */
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::binary_ivf::index<uint8_t, int64_t>& idx)
  -> cuvs::neighbors::binary_ivf::index<uint8_t, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   binary_ivf::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = binary_ivf::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   binary_ivf::extend(handle, dataset, no_opt, &index_empty);
 * @endcode
 *
 *
 * @param[in] handle
 * @param[in] new_vectors raft::device_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::device_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to index, to be overwritten in-place
 */
void extend(raft::resources const& handle,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::binary_ivf::index<uint8_t, int64_t>* idx);

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are adjusted to match the newly labeled data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   binary_ivf::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = binary_ivf::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = binary_ivf::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors raft::host_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::host_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] idx original index
 *
 * @return the constructed extended ivf-flat index
 */
auto extend(raft::resources const& handle,
            raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::binary_ivf::index<float, int64_t>& idx)
  -> cuvs::neighbors::binary_ivf::index<float, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   binary_ivf::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = binary_ivf::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   binary_ivf::extend(handle, dataset, no_opt, &index_empty);
 * @endcode
 *
 *
 * @param[in] handle
 * @param[in] new_vectors raft::host_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::host_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to index, to be overwritten in-place
 */
void extend(raft::resources const& handle,
            raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::binary_ivf::index<float, int64_t>* idx);

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are adjusted to match the newly labeled data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   binary_ivf::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = binary_ivf::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = binary_ivf::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors raft::host_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::host_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] idx original index
 *
 * @return the constructed extended ivf-flat index
 */
auto extend(raft::resources const& handle,
            raft::host_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::binary_ivf::index<half, int64_t>& idx)
  -> cuvs::neighbors::binary_ivf::index<half, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   binary_ivf::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = binary_ivf::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   binary_ivf::extend(handle, dataset, no_opt, &index_empty);
 * @endcode
 *
 *
 * @param[in] handle
 * @param[in] new_vectors raft::host_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::host_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to index, to be overwritten in-place
 */
void extend(raft::resources const& handle,
            raft::host_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::binary_ivf::index<half, int64_t>* idx);

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are adjusted to match the newly labeled data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   binary_ivf::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = binary_ivf::build(handle, dataset, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = binary_ivf::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors raft::host_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::host_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] idx original index
 *
 * @return the constructed extended ivf-flat index
 */
auto extend(raft::resources const& handle,
            raft::host_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::binary_ivf::index<int8_t, int64_t>& idx)
  -> cuvs::neighbors::binary_ivf::index<int8_t, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   binary_ivf::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = binary_ivf::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   binary_ivf::extend(handle, dataset, no_opt, &index_empty);
 * @endcode
 *
 *
 * @param[in] handle
 * @param[in] new_vectors raft::host_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::host_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to index, to be overwritten in-place
 */
void extend(raft::resources const& handle,
            raft::host_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::binary_ivf::index<int8_t, int64_t>* idx);

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are adjusted to match the newly labeled data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   binary_ivf::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = binary_ivf::build(handle, dataset, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = binary_ivf::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors raft::host_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::host_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] idx original index
 *
 * @return the constructed extended ivf-flat index
 */
auto extend(raft::resources const& handle,
            raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::binary_ivf::index<uint8_t, int64_t>& idx)
  -> cuvs::neighbors::binary_ivf::index<uint8_t, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   binary_ivf::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = binary_ivf::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   binary_ivf::extend(handle, dataset, no_opt, &index_empty);
 * @endcode
 *
 *
 * @param[in] handle
 * @param[in] new_vectors raft::host_matrix_view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices optional raft::host_vector_view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx pointer to index, to be overwritten in-place
 */
void extend(raft::resources const& handle,
            raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::binary_ivf::index<uint8_t, int64_t>* idx);
/**
 * @}
 */

/**
 * @defgroup binary_ivf_cpp_index_search IVF-Flat index search
 * @{
 */

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [binary_ivf::build](#binary_ivf::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`:
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   binary_ivf::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   binary_ivf::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   binary_ivf::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   binary_ivf::search(handle, search_params, index, queries3, out_inds3, out_dists3);
 *   ...
 * @endcode
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] index ivf-flat constructed index
 * @param[in] queries raft::device_matrix_view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors raft::device_matrix_view to the indices of the neighbors in the source
 * dataset [n_queries, k]
 * @param[out] distances raft::device_matrix_view to the distances to the selected neighbors
 * [n_queries, k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::binary_ivf::search_params& params,
            const cuvs::neighbors::binary_ivf::index<float, int64_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [binary_ivf::build](#binary_ivf::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`:
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   binary_ivf::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   binary_ivf::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   binary_ivf::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   binary_ivf::search(handle, search_params, index, queries3, out_inds3, out_dists3);
 *   ...
 * @endcode
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] index ivf-flat constructed index
 * @param[in] queries raft::device_matrix_view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors raft::device_matrix_view to the indices of the neighbors in the source
 * dataset [n_queries, k]
 * @param[out] distances raft::device_matrix_view to the distances to the selected neighbors
 * [n_queries, k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::binary_ivf::search_params& params,
            const cuvs::neighbors::binary_ivf::index<half, int64_t>& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});
/**
 * @brief Search ANN using the constructed index.
 *
 * See the [binary_ivf::build](#binary_ivf::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`:
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   binary_ivf::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   binary_ivf::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   binary_ivf::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   binary_ivf::search(handle, search_params, index, queries3, out_inds3, out_dists3);
 *   ...
 * @endcode
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] index ivf-flat constructed index
 * @param[in] queries raft::device_matrix_view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors raft::device_matrix_view to the indices of the neighbors in the source
 * dataset [n_queries, k]
 * @param[out] distances raft::device_matrix_view to the distances to the selected neighbors
 * [n_queries, k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::binary_ivf::search_params& params,
            const cuvs::neighbors::binary_ivf::index<int8_t, int64_t>& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [binary_ivf::build](#binary_ivf::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`:
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   binary_ivf::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   binary_ivf::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   binary_ivf::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   binary_ivf::search(handle, search_params, index, queries3, out_inds3, out_dists3);
 *   ...
 * @endcode
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] index ivf-flat constructed index
 * @param[in] queries raft::device_matrix_view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors raft::device_matrix_view to the indices of the neighbors in the source
 * dataset [n_queries, k]
 * @param[out] distances raft::device_matrix_view to the distances to the selected neighbors
 * [n_queries, k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::binary_ivf::search_params& params,
            const cuvs::neighbors::binary_ivf::index<uint8_t, int64_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @}
 */

/**
 * @defgroup binary_ivf_cpp_serialize IVF-Flat index serialize
 * @{
 */

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/binary_ivf.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = binary_ivf::build(...);`
 * cuvs::neighbors::binary_ivf::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-Flat index
 *
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::binary_ivf::index<int64_t>& index);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/binary_ivf.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * using T    = float; // data element type
 * using IdxT = int64_t; // type of the index
 * // create an empty index with `binary_ivf::index<IdxT> index(handle, index_params, dim);`
 * cuvs::neighbors::binary_ivf::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[in] index IVF-Flat index
 *
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::binary_ivf::index<int64_t>* index);
/**
 * @}
 */

/// \defgroup mg_cpp_index_build ANN MG index build

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<binary_ivf::index_params> index_params;
 * auto index = cuvs::neighbors::binary_ivf::build(clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-Flat MG index
 */
auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<binary_ivf::index_params>& index_params,
           raft::host_matrix_view<const float, int64_t, row_major> index_dataset)
  -> cuvs::neighbors::mg_index<binary_ivf::index<int64_t>, float, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<binary_ivf::index_params> index_params;
 * auto index = cuvs::neighbors::binary_ivf::build(clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-Flat MG index
 */
auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<binary_ivf::index_params>& index_params,
           raft::host_matrix_view<const int8_t, int64_t, row_major> index_dataset)
  -> cuvs::neighbors::mg_index<binary_ivf::index<int8_t, int64_t>, int8_t, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<binary_ivf::index_params> index_params;
 * auto index = cuvs::neighbors::binary_ivf::build(clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-Flat MG index
 */
auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<binary_ivf::index_params>& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, row_major> index_dataset)
  -> cuvs::neighbors::mg_index<binary_ivf::index<int64_t>, uint8_t, int64_t>;

/// \defgroup mg_cpp_index_extend ANN MG index extend

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<binary_ivf::index_params> index_params;
 * auto index = cuvs::neighbors::binary_ivf::build(clique, index_params, index_dataset);
 * cuvs::neighbors::binary_ivf::extend(clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& clique,
            cuvs::neighbors::mg_index<binary_ivf::index<int64_t>, float, int64_t>& index,
            raft::host_matrix_view<const float, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<binary_ivf::index_params> index_params;
 * auto index = cuvs::neighbors::binary_ivf::build(clique, index_params, index_dataset);
 * cuvs::neighbors::binary_ivf::extend(clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& clique,
            cuvs::neighbors::mg_index<binary_ivf::index<int64_t>, int8_t, int64_t>& index,
            raft::host_matrix_view<const int8_t, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<binary_ivf::index_params> index_params;
 * auto index = cuvs::neighbors::binary_ivf::build(clique, index_params, index_dataset);
 * cuvs::neighbors::binary_ivf::extend(clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& clique,
            cuvs::neighbors::mg_index<binary_ivf::index<int64_t>, uint8_t, int64_t>& index,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \defgroup mg_cpp_index_search ANN MG index search

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<binary_ivf::index_params> index_params;
 * auto index = cuvs::neighbors::binary_ivf::build(clique, index_params, index_dataset);
 * cuvs::neighbors::mg_search_params<binary_ivf::search_params> search_params;
 * cuvs::neighbors::binary_ivf::search(clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 *
 */
void search(const raft::resources& clique,
            const cuvs::neighbors::mg_index<binary_ivf::index<int64_t>, float, int64_t>& index,
            const cuvs::neighbors::mg_search_params<binary_ivf::search_params>& search_params,
            raft::host_matrix_view<const float, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<binary_ivf::index_params> index_params;
 * auto index = cuvs::neighbors::binary_ivf::build(clique, index_params, index_dataset);
 * cuvs::neighbors::mg_search_params<binary_ivf::search_params> search_params;
 * cuvs::neighbors::binary_ivf::search(clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 *
 */
void search(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<binary_ivf::index<int64_t>, uint8_t, int64_t>& index,
  const cuvs::neighbors::mg_search_params<binary_ivf::search_params>& search_params,
  raft::host_matrix_view<const uint8_t, int64_t, row_major> queries,
  raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances);

/// \defgroup mg_cpp_serialize ANN MG index serialization

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<binary_ivf::index_params> index_params;
 * auto index = cuvs::neighbors::binary_ivf::build(clique, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::binary_ivf::serialize(clique, index, filename);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<binary_ivf::index<int64_t>, uint8_t, int64_t>& index,
  const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<binary_ivf::index_params> index_params;
 * auto index = cuvs::neighbors::binary_ivf::build(clique, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::binary_ivf::serialize(clique, index, filename);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(
  const raft::resources& clique,
  const cuvs::neighbors::mg_index<binary_ivf::index<int64_t>, uint8_t, int64_t>& index,
  const std::string& filename);

/// \ingroup mg_cpp_deserialize
/**
 * @brief Deserializes an IVF-Flat multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<binary_ivf::index_params> index_params;
 * auto index = cuvs::neighbors::binary_ivf::build(clique, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::binary_ivf::serialize(clique, index, filename);
 * auto new_index = cuvs::neighbors::binary_ivf::deserialize<float, int64_t>(clique, filename);
 *
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] filename path to the file to be deserialized
 *
 */
template <typename IdxT>
auto deserialize(const raft::resources& clique, const std::string& filename)
  -> cuvs::neighbors::mg_index<binary_ivf::index<IdxT>, uint8_t, IdxT>;

/// \defgroup mg_cpp_distribute ANN MG local index distribution

/// \ingroup mg_cpp_distribute
/**
 * @brief Replicates a locally built and serialized IVF-Flat index to all GPUs to form a distributed
 * multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::binary_ivf::index_params index_params;
 * auto index = cuvs::neighbors::binary_ivf::build(clique, index_params, index_dataset);
 * const std::string filename = "local_index.cuvs";
 * cuvs::neighbors::binary_ivf::serialize(clique, filename, index);
 * auto new_index = cuvs::neighbors::binary_ivf::distribute<float, int64_t>(clique, filename);
 *
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] filename path to the file to be deserialized : a local index
 *
 */
template <typename IdxT>
auto distribute(const raft::resources& clique, const std::string& filename)
  -> cuvs::neighbors::mg_index<binary_ivf::index<IdxT>, uint8_t, IdxT>;

}  // namespace cuvs::neighbors::binary_ivf
