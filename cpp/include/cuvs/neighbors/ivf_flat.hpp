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

#include "ann_types.hpp"
#include <raft_runtime/neighbors/ivf_flat.hpp>

namespace cuvs::neighbors::ivf_flat {

struct index_params : ann::index_params {
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

  /** Build a raft IVF_FLAT index params from an existing cuvs IVF_FLAT index params. */
  operator raft::neighbors::ivf_flat::index_params() const
  {
    return raft::neighbors::ivf_flat::index_params{
      {
        .metric            = static_cast<raft::distance::DistanceType>((int)this->metric),
        .metric_arg        = this->metric_arg,
        .add_data_on_build = this->add_data_on_build,
      },
      .n_lists                        = n_lists,
      .kmeans_n_iters                 = kmeans_n_iters,
      .kmeans_trainset_fraction       = kmeans_trainset_fraction,
      .adaptive_centers               = adaptive_centers,
      .conservative_memory_allocation = conservative_memory_allocation};
  }
};

struct search_params : ann::search_params {
  /** The number of clusters to search. */
  uint32_t n_probes = 20;

  /** Build a raft IVF_FLAT search params from an existing cuvs IVF_FLAT search params. */
  operator raft::neighbors::ivf_flat::search_params() const
  {
    raft::neighbors::ivf_flat::search_params result = {{}, n_probes};
    return result;
  }
};

/**
 * @brief IVF-flat index.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 */
template <typename T, typename IdxT>
struct index : ann::index {
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

 public:
  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

  /** Construct an empty index. */
  index(raft::resources const& res, const index_params& params, uint32_t dim)
    : ann::index(),
      raft_index_(std::make_unique<raft::neighbors::ivf_flat::index<T, IdxT>>(
        res,
        static_cast<raft::distance::DistanceType>((int)params.metric),
        params.n_lists,
        params.adaptive_centers,
        params.conservative_memory_allocation,
        dim))
  {
  }

  /** Construct an empty index. It needs to be trained and then populated. */
  index(raft::resources const& res,
        cuvs::distance::DistanceType metric,
        uint32_t n_lists,
        bool adaptive_centers,
        bool conservative_memory_allocation,
        uint32_t dim)
    : ann::index(),
      raft_index_(res,
                  static_cast<raft::distance::DistanceType>((int)metric),
                  n_lists,
                  adaptive_centers,
                  conservative_memory_allocation,
                  dim)
  {
  }

  /** Build a cuvs IVF_FLAT index from an existing RAFT IVF_FLAT index. */
  index(raft::neighbors::ivf_flat::index<T, IdxT>&& raft_idx)
    : ann::index(),
      raft_index_(std::make_unique<raft::neighbors::ivf_flat::index<T, IdxT>>(std::move(raft_idx)))
  {
  }

  /**
   * Vectorized load/store size in elements, determines the size of interleaved data chunks.
   *
   * TODO: in theory, we can lift this to the template parameter and keep it at hardware maximum
   * possible value by padding the `dim` of the data https://github.com/rapidsai/raft/issues/711
   */
  [[nodiscard]] constexpr inline auto veclen() const noexcept -> uint32_t
  {
    return raft_index_->veclen();
  }
  /** Distance metric used for clustering. */
  [[nodiscard]] constexpr inline auto metric() const noexcept -> cuvs::distance::DistanceType
  {
    return static_cast<cuvs::distance::DistanceType>((int)raft_index_->metric());
  }
  /** Whether `centers()` change upon extending the index (ivf_pq::extend). */
  [[nodiscard]] constexpr inline auto adaptive_centers() const noexcept -> bool
  {
    return raft_index_->adaptive_centers();
  }
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
  inline auto list_sizes() noexcept -> raft::device_vector_view<uint32_t, uint32_t>
  {
    return raft_index_->list_sizes();
  }
  [[nodiscard]] inline auto list_sizes() const noexcept
    -> raft::device_vector_view<const uint32_t, uint32_t>
  {
    return raft_index_->list_sizes();
  }

  /** k-means cluster centers corresponding to the lists [n_lists, dim] */
  inline auto centers() noexcept -> raft::device_matrix_view<float, uint32_t, raft::row_major>
  {
    return raft_index_->centers();
  }
  [[nodiscard]] inline auto centers() const noexcept
    -> raft::device_matrix_view<const float, uint32_t, raft::row_major>
  {
    return raft_index_->centers();
  }

  /**
   * (Optional) Precomputed norms of the `centers` w.r.t. the chosen distance metric [n_lists].
   *
   * NB: this may be empty if the index is empty or if the metric does not require the center norms
   * calculation.
   */
  inline auto center_norms() noexcept -> std::optional<raft::device_vector_view<float, uint32_t>>
  {
    return raft_index_->center_norms();
  }
  [[nodiscard]] inline auto center_norms() const noexcept
    -> std::optional<raft::device_vector_view<const float, uint32_t>>
  {
    return raft_index_->center_norms();
  }

  /** Total length of the index. */
  [[nodiscard]] constexpr inline auto size() const noexcept -> IdxT { return raft_index_->size(); }
  /** Dimensionality of the data. */
  [[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t
  {
    return raft_index_->dim();
  }
  /** Number of clusters/inverted lists. */
  [[nodiscard]] constexpr inline auto n_lists() const noexcept -> uint32_t
  {
    return raft_index_->n_lists();
  }

  /** Pointers to the inverted lists (clusters) data  [n_lists]. */
  inline auto data_ptrs() noexcept -> raft::device_vector_view<T*, uint32_t>
  {
    return raft_index_->data_ptrs();
  }
  [[nodiscard]] inline auto data_ptrs() const noexcept
    -> raft::device_vector_view<T* const, uint32_t>
  {
    return raft_index_->data_ptrs();
  }

  /** Pointers to the inverted lists (clusters) indices  [n_lists]. */
  inline auto inds_ptrs() noexcept -> raft::device_vector_view<IdxT*, uint32_t>
  {
    return raft_index_->inds_ptrs();
  }
  [[nodiscard]] inline auto inds_ptrs() const noexcept
    -> raft::device_vector_view<IdxT* const, uint32_t>
  {
    return raft_index_->inds_ptrs();
  }
  /**
   * Whether to use convervative memory allocation when extending the list (cluster) data
   * (see index_params.conservative_memory_allocation).
   */
  [[nodiscard]] constexpr inline auto conservative_memory_allocation() const noexcept -> bool
  {
    return raft_index_->conservative_memory_allocation();
  }

  /** Lists' data and indices. */
  inline auto lists() noexcept { return raft_index_->lists(); }
  [[nodiscard]] inline auto lists() const noexcept { return raft_index_->lists(); }

  auto get_raft_index() const -> const raft::neighbors::ivf_flat::index<T, IdxT>*
  {
    return raft_index_.get();
  }
  auto get_raft_index() -> raft::neighbors::ivf_flat::index<T, IdxT>* { return raft_index_.get(); }

 private:
  std::unique_ptr<raft::neighbors::ivf_flat::index<T, IdxT>> raft_index_;
};

#define CUVS_IVF_FLAT(T, IdxT)                                                       \
  auto build(raft::resources const& handle,                                          \
             const cuvs::neighbors::ivf_flat::index_params& params,                  \
             raft::device_matrix_view<const T, IdxT, raft::row_major> dataset)       \
    ->cuvs::neighbors::ivf_flat::index<T, IdxT>;                                     \
                                                                                     \
  void build(raft::resources const& handle,                                          \
             const cuvs::neighbors::ivf_flat::index_params& params,                  \
             raft::device_matrix_view<const T, IdxT, raft::row_major> dataset,       \
             cuvs::neighbors::ivf_flat::index<T, IdxT>& idx);                        \
                                                                                     \
  auto extend(raft::resources const& handle,                                         \
              raft::device_matrix_view<const T, IdxT, raft::row_major> new_vectors,  \
              std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices, \
              const cuvs::neighbors::ivf_flat::index<T, IdxT>& orig_index)           \
    ->cuvs::neighbors::ivf_flat::index<T, IdxT>;                                     \
                                                                                     \
  void extend(raft::resources const& handle,                                         \
              raft::device_matrix_view<const T, IdxT, raft::row_major> new_vectors,  \
              std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices, \
              cuvs::neighbors::ivf_flat::index<T, IdxT>* idx);                       \
                                                                                     \
  void search(raft::resources const& handle,                                         \
              const cuvs::neighbors::ivf_flat::search_params& params,                \
              cuvs::neighbors::ivf_flat::index<T, IdxT>& index,                      \
              raft::device_matrix_view<const T, IdxT, raft::row_major> queries,      \
              raft::device_matrix_view<IdxT, IdxT, raft::row_major> neighbors,       \
              raft::device_matrix_view<float, IdxT, raft::row_major> distances);     \
                                                                                     \
  void serialize_file(raft::resources const& handle,                                 \
                      const std::string& filename,                                   \
                      const cuvs::neighbors::ivf_flat::index<T, IdxT>& index);       \
                                                                                     \
  void deserialize_file(raft::resources const& handle,                               \
                        const std::string& filename,                                 \
                        cuvs::neighbors::ivf_flat::index<T, IdxT>* index);           \
                                                                                     \
  void serialize(raft::resources const& handle,                                      \
                 std::string& str,                                                   \
                 const cuvs::neighbors::ivf_flat::index<T, IdxT>& index);            \
                                                                                     \
  void deserialize(raft::resources const& handle,                                    \
                   const std::string& str,                                           \
                   cuvs::neighbors::ivf_flat::index<T, IdxT>* index);

CUVS_IVF_FLAT(float, int64_t);
CUVS_IVF_FLAT(int8_t, int64_t);
CUVS_IVF_FLAT(uint8_t, int64_t);

#undef CUVS_IVF_FLAT

}  // namespace cuvs::neighbors::ivf_flat