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

#include "common.hpp"
#include <cstdint>
#include <cuvs/neighbors/common.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>

namespace cuvs::neighbors::ivf_flat {
/**
 * @defgroup ivf_flat_cpp_index_params IVF-Flat index build parameters
 * @{
 */

/** Size of the interleaved group (see `index::data` description). */
constexpr static uint32_t kIndexGroupSize = 32;

struct index_params : cuvs::neighbors::index_params {
  /** The number of inverted lists (clusters) */
  uint32_t n_lists = 1024;
  /** The number of iterations searching for kmeans centers (index building). */
  uint32_t kmeans_n_iters = 20;
  /** The fraction of data to use during iterative kmeans building. */
  double kmeans_trainset_fraction = 0.5;
  /**
   * By default (adaptive_centers = false), the cluster centers are trained in `ivf_flat::build`,
   * and never modified in `ivf_flat::extend`. As a result, you may need to retrain the index
   * from scratch after invoking (`ivf_flat::extend`) a few times with new data, the distribution of
   * which is no longer representative of the original training set.
   *
   * The alternative behavior (adaptive_centers = true) is to update the cluster centers for new
   * data when it is added. In this case, `index.centers()` are always exactly the centroids of the
   * data in the corresponding clusters. The drawback of this behavior is that the centroids depend
   * on the order of adding new data (through the classification of the added data); that is,
   * `index.centers()` "drift" together with the changing distribution of the newly added data.
   */
  bool adaptive_centers = false;
  /**
   * By default, the algorithm allocates more space than necessary for individual clusters
   * (`list_data`). This allows to amortize the cost of memory allocation and reduce the number of
   * data copies during repeated calls to `extend` (extending the database).
   *
   * The alternative is the conservative allocation behavior; when enabled, the algorithm always
   * allocates the minimum amount of memory required to store the given number of records. Set this
   * flag to `true` if you prefer to use as little GPU memory for the database as possible.
   */
  bool conservative_memory_allocation = false;
  /**
   * Whether to add the dataset content to the index, i.e.:
   *
   *  - `true` means the index is filled with the dataset vectors and ready to search after calling
   * `build`.
   *  - `false` means `build` only trains the underlying model (e.g. quantizer or clustering), but
   * the index is left empty; you'd need to call `extend` on the index afterwards to populate it.
   */
  bool add_data_on_build = true;
};
/**
 * @}
 */

/**
 * @defgroup ivf_flat_cpp_search_params IVF-Flat index search parameters
 * @{
 */
struct search_params : cuvs::neighbors::search_params {
  /** The number of clusters to search. */
  uint32_t n_probes = 20;
};

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
 * @defgroup ivf_flat_cpp_index IVF-Flat index
 * @{
 */
/**
 * @brief IVF-flat index.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 */
template <typename T, typename IdxT>
struct index : cuvs::neighbors::index {
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
        cuvs::distance::DistanceType metric,
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

  /** Whether `centers()` change upon extending the index (ivf_flat::extend). */
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
  raft::device_vector_view<T*, uint32_t> data_ptrs() noexcept;
  raft::device_vector_view<T* const, uint32_t> data_ptrs() const noexcept;

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
  std::vector<std::shared_ptr<list_data<T, IdxT>>>& lists() noexcept;
  const std::vector<std::shared_ptr<list_data<T, IdxT>>>& lists() const noexcept;

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
  std::vector<std::shared_ptr<list_data<T, IdxT>>> lists_;
  raft::device_vector<uint32_t, uint32_t> list_sizes_;
  raft::device_matrix<float, uint32_t, raft::row_major> centers_;
  std::optional<raft::device_vector<float, uint32_t>> center_norms_;

  // Computed members
  raft::device_vector<T*, uint32_t> data_ptrs_;
  raft::device_vector<IdxT*, uint32_t> inds_ptrs_;
  raft::host_vector<IdxT, uint32_t> accum_sorted_sizes_;

  static auto calculate_veclen(uint32_t dim) -> uint32_t
  {
    // TODO: consider padding the dimensions and fixing veclen to its maximum possible value as a
    // template parameter (https://github.com/rapidsai/raft/issues/711)

    // NOTE: keep this consistent with the select_interleaved_scan_kernel logic
    // in detail/ivf_flat_interleaved_scan-inl.cuh.
    uint32_t veclen = std::max<uint32_t>(1, 16 / sizeof(T));
    if (dim % veclen != 0) { veclen = 1; }
    return veclen;
  }
};
/**
 * @}
 */

/**
 * @defgroup ivf_flat_cpp_index_build IVF-Flat index build
 * @{
 */
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
 *   ivf_flat::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_flat::build(handle, dataset, index_params);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-flat index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_flat::index<float, int64_t>;

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
 *   ivf_flat::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   ivf_flat::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_flat::build(handle, dataset, index_params, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::device_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_flat::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_flat::index<float, int64_t>& idx);

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
 *   ivf_flat::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_flat::build(handle, dataset, index_params);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-flat index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_flat::index<int8_t, int64_t>;

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
 *   ivf_flat::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   ivf_flat::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_flat::build(handle, dataset, index_params, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::device_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_flat::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& idx);

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
 *   ivf_flat::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_flat::build(handle, dataset, index_params);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-flat index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>;

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
 *   ivf_flat::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   ivf_flat::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_flat::build(handle, dataset, index_params, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::device_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_flat::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& idx);

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
 *   ivf_flat::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_flat::build(handle, dataset, index_params);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-flat index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_flat::index<float, int64_t>;

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
 *   ivf_flat::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   ivf_flat::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_flat::build(handle, dataset, index_params, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_flat::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_flat::index<float, int64_t>& idx);

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
 *   ivf_flat::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_flat::build(handle, dataset, index_params);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a host pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-flat index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_flat::index<int8_t, int64_t>;

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
 *   ivf_flat::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   ivf_flat::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_flat::build(handle, dataset, index_params, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_flat::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& idx);

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
 *   ivf_flat::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_flat::build(handle, dataset, index_params);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a host pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-flat index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>;

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
 *   ivf_flat::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   ivf_flat::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_flat::build(handle, dataset, index_params, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_flat::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& idx);
/**
 * @}
 */

/**
 * @defgroup ivf_flat_cpp_index_extend IVF-Flat index extend
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
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_flat::extend(handle, new_vectors, no_op, index_empty);
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
            const cuvs::neighbors::ivf_flat::index<float, int64_t>& idx)
  -> cuvs::neighbors::ivf_flat::index<float, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_flat::extend(handle, dataset, no_opt, &index_empty);
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
            cuvs::neighbors::ivf_flat::index<float, int64_t>* idx);

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
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, dataset, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_flat::extend(handle, new_vectors, no_op, index_empty);
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
            const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& idx)
  -> cuvs::neighbors::ivf_flat::index<int8_t, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_flat::extend(handle, dataset, no_opt, &index_empty);
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
            cuvs::neighbors::ivf_flat::index<int8_t, int64_t>* idx);

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
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, dataset, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_flat::extend(handle, new_vectors, no_op, index_empty);
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
            const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& idx)
  -> cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_flat::extend(handle, dataset, no_opt, &index_empty);
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
            cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>* idx);

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
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_flat::extend(handle, new_vectors, no_op, index_empty);
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
            const cuvs::neighbors::ivf_flat::index<float, int64_t>& idx)
  -> cuvs::neighbors::ivf_flat::index<float, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_flat::extend(handle, dataset, no_opt, &index_empty);
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
            cuvs::neighbors::ivf_flat::index<float, int64_t>* idx);

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
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, dataset, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_flat::extend(handle, new_vectors, no_op, index_empty);
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
            const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& idx)
  -> cuvs::neighbors::ivf_flat::index<int8_t, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_flat::extend(handle, dataset, no_opt, &index_empty);
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
            cuvs::neighbors::ivf_flat::index<int8_t, int64_t>* idx);

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
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, dataset, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_flat::extend(handle, new_vectors, no_op, index_empty);
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
            const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& idx)
  -> cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>;

/**
 * @brief Extend the index in-place with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_flat::extend(handle, dataset, no_opt, &index_empty);
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
            cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>* idx);
/**
 * @}
 */

/**
 * @defgroup ivf_flat_cpp_index_search IVF-Flat index search
 * @{
 */

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [ivf_flat::build](#ivf_flat::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`:
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   ivf_flat::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_flat::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   ivf_flat::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   ivf_flat::search(handle, search_params, index, queries3, out_inds3, out_dists3);
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
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_flat::search_params& params,
            cuvs::neighbors::ivf_flat::index<float, int64_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [ivf_flat::build](#ivf_flat::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`:
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   ivf_flat::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_flat::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   ivf_flat::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   ivf_flat::search(handle, search_params, index, queries3, out_inds3, out_dists3);
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
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_flat::search_params& params,
            cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [ivf_flat::build](#ivf_flat::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`:
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   ivf_flat::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_flat::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   ivf_flat::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   ivf_flat::search(handle, search_params, index, queries3, out_inds3, out_dists3);
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
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_flat::search_params& params,
            cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/**
 * @brief Search ANN using the constructed index with the given filter.
 *
 * See the [ivf_flat::build](#ivf_flat::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`.
 * The exact size of the temporary buffer depends on multiple factors and is an implementation
 * detail. However, you can safely specify a small initial size for the memory pool, so that only a
 * few allocations happen to grow it during the first invocations of the `search`.
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] idx ivf-flat constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter a device bitset filter function that greenlights samples for a given
 * query.
 */
void search_with_filtering(
  raft::resources const& handle,
  const search_params& params,
  index<float, int64_t>& idx,
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> sample_filter);

/**
 * @brief Search ANN using the constructed index with the given filter.
 *
 * See the [ivf_flat::build](#ivf_flat::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`.
 * The exact size of the temporary buffer depends on multiple factors and is an implementation
 * detail. However, you can safely specify a small initial size for the memory pool, so that only a
 * few allocations happen to grow it during the first invocations of the `search`.
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] idx ivf-flat constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter a device bitset filter function that greenlights samples for a given
 * query.
 */
void search_with_filtering(
  raft::resources const& handle,
  const search_params& params,
  index<int8_t, int64_t>& idx,
  raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> sample_filter);

/**
 * @brief Search ANN using the constructed index with the given filter.
 *
 * See the [ivf_flat::build](#ivf_flat::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`.
 * The exact size of the temporary buffer depends on multiple factors and is an implementation
 * detail. However, you can safely specify a small initial size for the memory pool, so that only a
 * few allocations happen to grow it during the first invocations of the `search`.
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] idx ivf-flat constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter a device bitset filter function that greenlights samples for a given
 * query.
 */
void search_with_filtering(
  raft::resources const& handle,
  const search_params& params,
  index<uint8_t, int64_t>& idx,
  raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> sample_filter);
/**
 * @}
 */

/**
 * @defgroup ivf_flat_cpp_serialize IVF-Flat index serialize
 * @{
 */

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = ivf_flat::build(...);`
 * cuvs::neighbors::ivf_flat::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-Flat index
 *
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::ivf_flat::index<float, int64_t>& index);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * using T    = float; // data element type
 * using IdxT = int64_t; // type of the index
 * // create an empty index with `ivf_flat::index<T, IdxT> index(handle, index_params, dim);`
 * cuvs::neighbors::ivf_flat::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[in] index IVF-Flat index
 *
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_flat::index<float, int64_t>* index);

/**
 * Write the index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = ivf_flat::build(...);`
 * cuvs::neighbors::ivf_flat::serialize(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index IVF-Flat index
 *
 */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::ivf_flat::index<float, int64_t>& index);

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 * using T    = float; // data element type
 * using IdxT = int64_t; // type of the index
 * // create an empty index with `ivf_flat::index<T, IdxT> index(handle, index_params, dim);`
 * cuvs::neighbors::ivf_flat::deserialize(handle, is, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[in] index IVF-Flat index
 *
 */
void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::ivf_flat::index<float, int64_t>* index);

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = ivf_flat::build(...);`
 * cuvs::neighbors::ivf_flat::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-Flat index
 *
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& index);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * using T    = float; // data element type
 * using IdxT = int64_t; // type of the index
 * // create an empty index with `ivf_flat::index<T, IdxT> index(handle, index_params, dim);`
 * cuvs::neighbors::ivf_flat::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[in] index IVF-Flat index
 *
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_flat::index<int8_t, int64_t>* index);

/**
 * Write the index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = ivf_flat::build(...);`
 * cuvs::neighbors::ivf_flat::serialize(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index IVF-Flat index
 *
 */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& index);

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 * using T    = float; // data element type
 * using IdxT = int64_t; // type of the index
 * // create an empty index with `ivf_flat::index<T, IdxT> index(handle, index_params, dim);`
 * cuvs::neighbors::ivf_flat::deserialize(handle, is, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[in] index IVF-Flat index
 *
 */
void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::ivf_flat::index<int8_t, int64_t>* index);

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = ivf_flat::build(...);`
 * cuvs::neighbors::ivf_flat::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-Flat index
 *
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& index);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * using T    = float; // data element type
 * using IdxT = int64_t; // type of the index
 * // create an empty index with ivf_flat::index<T, IdxT> index(handle, index_params, dim);`
 * cuvs::neighbors::ivf_flat::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[in] index IVF-Flat index
 *
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>* index);

/**
 * Write the index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = ivf_flat::build(...);`
 * cuvs::neighbors::ivf_flat::serialize(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index IVF-Flat index
 *
 */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& index);

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/ivf_flat.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 * using T    = float; // data element type
 * using IdxT = int64_t; // type of the index
 * // create an empty index with `ivf_flat::index<T, IdxT> index(handle, index_params, dim);`
 * cuvs::neighbors::ivf_flat::deserialize(handle, is, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[in] index IVF-Flat index
 *
 */
void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>* index);

/**
 * @}
 */

namespace helpers {

/**
 * @defgroup ivf_flat_helpers Helper functions for IVF Flat
 * @{
 */

namespace codepacker {

/**
 * Write flat codes into an existing list by the given offset.
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_vec).
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data  = index.lists()[label]->data.view();
 *   // allocate the buffer for the input codes
 *   auto codes = raft::make_device_matrix<float>(res, n_vec, index.dim());
 *   ... prepare n_vecs to pack into the list in codes ...
 *   // write codes into the list starting from the 42nd position
 *   ivf_flat::helpers::codepacker::pack(
 *       res, make_const_mdspan(codes.view()), index.veclen(), 42, list_data);
 * @endcode
 *
 * @param[in] res
 * @param[in] codes flat codes [n_vec, dim]
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset how many records to skip before writing the data into the list
 * @param[inout] list_data block to write into
 */
void pack(raft::resources const& res,
          raft::device_matrix_view<const float, uint32_t, raft::row_major> codes,
          uint32_t veclen,
          uint32_t offset,
          raft::device_mdspan<float,
                              typename list_spec<uint32_t, float, int64_t>::list_extents,
                              raft::row_major> list_data);

/**
 * Write flat codes into an existing list by the given offset.
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_vec).
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data  = index.lists()[label]->data.view();
 *   // allocate the buffer for the input codes
 *   auto codes = raft::make_device_matrix<int8_t>(res, n_vec, index.dim());
 *   ... prepare n_vecs to pack into the list in codes ...
 *   // write codes into the list starting from the 42nd position
 *   ivf_flat::helpers::codepacker::pack(
 *       res, make_const_mdspan(codes.view()), index.veclen(), 42, list_data);
 * @endcode
 *
 * @param[in] res
 * @param[in] codes flat codes [n_vec, dim]
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset how many records to skip before writing the data into the list
 * @param[inout] list_data block to write into
 */
void pack(raft::resources const& res,
          raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> codes,
          uint32_t veclen,
          uint32_t offset,
          raft::device_mdspan<int8_t,
                              typename list_spec<uint32_t, int8_t, int64_t>::list_extents,
                              raft::row_major> list_data);

/**
 * Write flat codes into an existing list by the given offset.
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_vec).
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data  = index.lists()[label]->data.view();
 *   // allocate the buffer for the input codes
 *   auto codes = raft::make_device_matrix<uint8_t>(res, n_vec, index.dim());
 *   ... prepare n_vecs to pack into the list in codes ...
 *   // write codes into the list starting from the 42nd position
 *   ivf_flat::helpers::codepacker::pack(
 *       res, make_const_mdspan(codes.view()), index.veclen(), 42, list_data);
 * @endcode
 *
 * @param[in] res
 * @param[in] codes flat codes [n_vec, dim]
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset how many records to skip before writing the data into the list
 * @param[inout] list_data block to write into
 */
void pack(raft::resources const& res,
          raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
          uint32_t veclen,
          uint32_t offset,
          raft::device_mdspan<uint8_t,
                              typename list_spec<uint32_t, uint8_t, int64_t>::list_extents,
                              raft::row_major> list_data);

/**
 * @brief Unpack `n_take` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`.
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data = index.lists()[label]->data.view();
 *   // allocate the buffer for the output
 *   uint32_t n_take = 4;
 *   auto codes = raft::make_device_matrix<float>(res, n_take, index.dim());
 *   uint32_t offset = 0;
 *   // unpack n_take elements from the list
 *   ivf_fat::helpers::codepacker::unpack(res, list_data, index.veclen(), offset, codes.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] list_data block to read from
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset
 *   How many records in the list to skip.
 * @param[inout] codes
 *   the destination buffer [n_take, index.dim()].
 *   The length `n_take` defines how many records to unpack,
 *   it must be <= the list size.
 */
void unpack(raft::resources const& res,
            raft::device_mdspan<const float,
                                typename list_spec<uint32_t, float, int64_t>::list_extents,
                                raft::row_major> list_data,
            uint32_t veclen,
            uint32_t offset,
            raft::device_matrix_view<float, uint32_t, raft::row_major> codes);

/**
 * @brief Unpack `n_take` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`.
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data = index.lists()[label]->data.view();
 *   // allocate the buffer for the output
 *   uint32_t n_take = 4;
 *   auto codes = raft::make_device_matrix<int8_t>(res, n_take, index.dim());
 *   uint32_t offset = 0;
 *   // unpack n_take elements from the list
 *   ivf_fat::helpers::codepacker::unpack(res, list_data, index.veclen(), offset, codes.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] list_data block to read from
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset
 *   How many records in the list to skip.
 * @param[inout] codes
 *   the destination buffer [n_take, index.dim()].
 *   The length `n_take` defines how many records to unpack,
 *   it must be <= the list size.
 */
void unpack(raft::resources const& res,
            raft::device_mdspan<const int8_t,
                                typename list_spec<uint32_t, int8_t, int64_t>::list_extents,
                                raft::row_major> list_data,
            uint32_t veclen,
            uint32_t offset,
            raft::device_matrix_view<int8_t, uint32_t, raft::row_major> codes);

/**
 * @brief Unpack `n_take` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`.
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data = index.lists()[label]->data.view();
 *   // allocate the buffer for the output
 *   uint32_t n_take = 4;
 *   auto codes = raft::make_device_matrix<uint8_t>(res, n_take, index.dim());
 *   uint32_t offset = 0;
 *   // unpack n_take elements from the list
 *   ivf_fat::helpers::codepacker::unpack(res, list_data, index.veclen(), offset, codes.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] list_data block to read from
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset
 *   How many records in the list to skip.
 * @param[inout] codes
 *   the destination buffer [n_take, index.dim()].
 *   The length `n_take` defines how many records to unpack,
 *   it must be <= the list size.
 */
void unpack(raft::resources const& res,
            raft::device_mdspan<const uint8_t,
                                typename list_spec<uint32_t, uint8_t, int64_t>::list_extents,
                                raft::row_major> list_data,
            uint32_t veclen,
            uint32_t offset,
            raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> codes);

/**
 * Write one flat code into a block by the given offset. The offset indicates the id of the record
 * in the list. This function interleaves the code and is intended to later copy the interleaved
 * codes over to the IVF list on device. NB: no memory allocation happens here; the block must fit
 * the record (offset + 1).
 *
 * @param[in] flat_code input flat code
 * @param[out] block block of memory to write interleaved codes to
 * @param[in] dim dimension of the flat code
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset how many records to skip before writing the data into the list
 */
void pack_1(const float* flat_code, float* block, uint32_t dim, uint32_t veclen, uint32_t offset);

/**
 * Write one flat code into a block by the given offset. The offset indicates the id of the record
 * in the list. This function interleaves the code and is intended to later copy the interleaved
 * codes over to the IVF list on device. NB: no memory allocation happens here; the block must fit
 * the record (offset + 1).
 *
 * @param[in] flat_code input flat code
 * @param[out] block block of memory to write interleaved codes to
 * @param[in] dim dimension of the flat code
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset how many records to skip before writing the data into the list
 */
void pack_1(const int8_t* flat_code, int8_t* block, uint32_t dim, uint32_t veclen, uint32_t offset);

/**
 * Write one flat code into a block by the given offset. The offset indicates the id of the record
 * in the list. This function interleaves the code and is intended to later copy the interleaved
 * codes over to the IVF list on device. NB: no memory allocation happens here; the block must fit
 * the record (offset + 1).
 *
 * @param[in] flat_code input flat code
 * @param[out] block block of memory to write interleaved codes to
 * @param[in] dim dimension of the flat code
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset how many records to skip before writing the data into the list
 */
void pack_1(
  const uint8_t* flat_code, uint8_t* block, uint32_t dim, uint32_t veclen, uint32_t offset);

/**
 * Unpack 1 record of a single list (cluster) in the index to fetch the flat code. The offset
 * indicates the id of the record. This function fetches one flat code from an interleaved code.
 *
 * @param[in] block interleaved block. The block can be thought of as the whole inverted list in
 * interleaved format.
 * @param[out] flat_code output flat code
 * @param[in] dim dimension of the flat code
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset fetch the flat code by the given offset
 */
void unpack_1(const float* block, float* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset);

/**
 * Unpack 1 record of a single list (cluster) in the index to fetch the flat code. The offset
 * indicates the id of the record. This function fetches one flat code from an interleaved code.
 *
 * @param[in] block interleaved block. The block can be thought of as the whole inverted list in
 * interleaved format.
 * @param[out] flat_code output flat code
 * @param[in] dim dimension of the flat code
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset fetch the flat code by the given offset
 */
void unpack_1(
  const int8_t* block, int8_t* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset);

/**
 * Unpack 1 record of a single list (cluster) in the index to fetch the flat code. The offset
 * indicates the id of the record. This function fetches one flat code from an interleaved code.
 *
 * @param[in] block interleaved block. The block can be thought of as the whole inverted list in
 * interleaved format.
 * @param[out] flat_code output flat code
 * @param[in] dim dimension of the flat code
 * @param[in] veclen size of interleaved data chunks
 * @param[in] offset fetch the flat code by the given offset
 */
void unpack_1(
  const uint8_t* block, uint8_t* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset);

}  // namespace codepacker

/**
 * @brief Public helper API to reset the data and indices ptrs, and the list sizes. Useful for
 * externally modifying the index without going through the build stage. The data and indices of the
 * IVF lists will be lost.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // initialize an empty index
 *   ivf_flat::index<float, int64_t> index(res, index_params, D);
 *   // reset the index's state and list sizes
 *   ivf_flat::helpers::reset_index(res, &index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-Flat index
 */
void reset_index(const raft::resources& res, index<float, int64_t>* index);

/**
 * @brief Public helper API to reset the data and indices ptrs, and the list sizes. Useful for
 * externally modifying the index without going through the build stage. The data and indices of the
 * IVF lists will be lost.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // initialize an empty index
 *   ivf_flat::index<int8_t, int64_t> index(res, index_params, D);
 *   // reset the index's state and list sizes
 *   ivf_flat::helpers::reset_index(res, &index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-Flat index
 */
void reset_index(const raft::resources& res, index<int8_t, int64_t>* index);

/**
 * @brief Public helper API to reset the data and indices ptrs, and the list sizes. Useful for
 * externally modifying the index without going through the build stage. The data and indices of the
 * IVF lists will be lost.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // initialize an empty index
 *   ivf_flat::index<uint8_t, int64_t> index(res, index_params, D);
 *   // reset the index's state and list sizes
 *   ivf_flat::helpers::reset_index(res, &index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-Flat index
 */
void reset_index(const raft::resources& res, index<uint8_t, int64_t>* index);

/**
 * @brief Helper exposing the re-computation of list sizes and related arrays if IVF lists have been
 * modified externally.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   raft::resources res;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // initialize an empty index
 *   ivf_pq::index<int64_t> index(res, index_params, D);
 *   ivf_pq::helpers::reset_index(res, &index);
 *   // resize the first IVF list to hold 5 records
 *   auto spec = list_spec<uint32_t, int64_t>{
 *     index->pq_bits(), index->pq_dim(), index->conservative_memory_allocation()};
 *   uint32_t new_size = 5;
 *   ivf::resize_list(res, list, spec, new_size, 0);
 *   raft::update_device(index.list_sizes(), &new_size, 1, stream);
 *   // recompute the internal state of the index
 *   ivf_pq::helpers::recompute_internal_state(res, index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-PQ index
 */
void recompute_internal_state(const raft::resources& res, index<float, int64_t>* index);

/**
 * @brief Helper exposing the re-computation of list sizes and related arrays if IVF lists have been
 * modified externally.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   raft::resources res;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // initialize an empty index
 *   ivf_pq::index<int64_t> index(res, index_params, D);
 *   ivf_pq::helpers::reset_index(res, &index);
 *   // resize the first IVF list to hold 5 records
 *   auto spec = list_spec<uint32_t, int64_t>{
 *     index->pq_bits(), index->pq_dim(), index->conservative_memory_allocation()};
 *   uint32_t new_size = 5;
 *   ivf::resize_list(res, list, spec, new_size, 0);
 *   raft::update_device(index.list_sizes(), &new_size, 1, stream);
 *   // recompute the internal state of the index
 *   ivf_pq::helpers::recompute_internal_state(res, index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-PQ index
 */
void recompute_internal_state(const raft::resources& res, index<int8_t, int64_t>* index);

/**
 * @brief Helper exposing the re-computation of list sizes and related arrays if IVF lists have been
 * modified externally.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   raft::resources res;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // initialize an empty index
 *   ivf_pq::index<int64_t> index(res, index_params, D);
 *   ivf_pq::helpers::reset_index(res, &index);
 *   // resize the first IVF list to hold 5 records
 *   auto spec = list_spec<uint32_t, int64_t>{
 *     index->pq_bits(), index->pq_dim(), index->conservative_memory_allocation()};
 *   uint32_t new_size = 5;
 *   ivf::resize_list(res, list, spec, new_size, 0);
 *   raft::update_device(index.list_sizes(), &new_size, 1, stream);
 *   // recompute the internal state of the index
 *   ivf_pq::helpers::recompute_internal_state(res, index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-Flat index
 */
void recompute_internal_state(const raft::resources& res, index<uint8_t, int64_t>* index);
/**
 * @}
 */

}  // namespace helpers
}  // namespace cuvs::neighbors::ivf_flat
