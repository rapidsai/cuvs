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
#include <raft/neighbors/ivf_flat_types.hpp>

namespace cuvs::neighbors::ivf_flat {
/**
 * @defgroup ivf_flat_cpp_index_params IVF-Flat index build parameters
 * @{
 */
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
/**
 * @}
 */

/**
 * @defgroup ivf_flat_cpp_search_params IVF-Flat index search parameters
 * @{
 */
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
 * @}
 */

/**
 * @defgroup ivf_flat_cpp_index IVF-Flat index
 * @{
 */
template <typename T, typename IdxT>
struct index : ann::index {
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

 public:
  index(const index&)            = delete;
  index(index&&)                 = default;
  index& operator=(const index&) = delete;
  index& operator=(index&&)      = default;
  ~index()                       = default;
  /** Construct an empty index. It needs to be trained and then populated. */
  index(raft::resources const& res, const index_params& params, uint32_t dim);
  /** Construct an empty index. It needs to be trained and then populated. */
  index(raft::resources const& res,
        cuvs::distance::DistanceType metric,
        uint32_t n_lists,
        bool adaptive_centers,
        bool conservative_memory_allocation,
        uint32_t dim);
  index(raft::neighbors::ivf_flat::index<T, IdxT>&& raft_idx);

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

  /** Lists' data and indices. */
  std::vector<std::shared_ptr<raft::neighbors::ivf_flat::list_data<T, IdxT>>>& lists() noexcept;
  const std::vector<std::shared_ptr<raft::neighbors::ivf_flat::list_data<T, IdxT>>>& lists()
    const noexcept;

  // Get pointer to underlying RAFT index, not meant to be used outside of cuVS
  inline raft::neighbors::ivf_flat::index<T, IdxT>* get_raft_index() noexcept
  {
    return raft_index_.get();
  }
  inline const raft::neighbors::ivf_flat::index<T, IdxT>* get_raft_index() const noexcept
  {
    return raft_index_.get();
  }

 private:
  std::unique_ptr<raft::neighbors::ivf_flat::index<T, IdxT>> raft_index_;
};
/**
 * @}
 */

/**
 * @defgroup ivf_flat_cpp_index_build IVF-Flat index build
 * @{
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_flat::index<float, int64_t>;

void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_flat::index<float, int64_t>& idx);

auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& params,
           raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_flat::index<int8_t, int64_t>;

void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& params,
           raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& idx);

auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>;

void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_flat::index_params& params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& idx);
/**
 * @}
 */

/**
 * @defgroup ivf_flat_cpp_index_extend IVF-Flat index extend
 * @{
 */
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_flat::index<float, int64_t>& orig_index)
  -> cuvs::neighbors::ivf_flat::index<float, int64_t>;

void extend(raft::resources const& handle,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_flat::index<float, int64_t>* idx);

auto extend(raft::resources const& handle,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& orig_index)
  -> cuvs::neighbors::ivf_flat::index<int8_t, int64_t>;

void extend(raft::resources const& handle,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_flat::index<int8_t, int64_t>* idx);

auto extend(raft::resources const& handle,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& orig_index)
  -> cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>;

void extend(raft::resources const& handle,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>* idx);
/**
 * @}
 */

/**
 * @defgroup ivf_flat_cpp_index_search IVF-Flat index search
 * @{
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_flat::search_params& params,
            cuvs::neighbors::ivf_flat::index<float, int64_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_flat::search_params& params,
            cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_flat::search_params& params,
            cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);
/**
 * @}
 */

/**
 * @defgroup ivf_flat_cpp_serialize IVF-Flat index serialize
 * @{
 */
void serialize_file(raft::resources const& handle,
                    const std::string& filename,
                    const cuvs::neighbors::ivf_flat::index<float, int64_t>& index);

void serialize_file(raft::resources const& handle,
                    const std::string& filename,
                    const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& index);

void serialize_file(raft::resources const& handle,
                    const std::string& filename,
                    const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& index);

void deserialize_file(raft::resources const& handle,
                      const std::string& filename,
                      cuvs::neighbors::ivf_flat::index<int8_t, int64_t>* index);

void deserialize_file(raft::resources const& handle,
                      const std::string& filename,
                      cuvs::neighbors::ivf_flat::index<float, int64_t>* index);

void deserialize_file(raft::resources const& handle,
                      const std::string& filename,
                      cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>* index);

void serialize(raft::resources const& handle,
               std::string& str,
               const cuvs::neighbors::ivf_flat::index<float, int64_t>& index);

void serialize(raft::resources const& handle,
               std::string& str,
               const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& index);

void serialize(raft::resources const& handle,
               std::string& str,
               const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& index);

void deserialize(raft::resources const& handle,
                 const std::string& str,
                 cuvs::neighbors::ivf_flat::index<float, int64_t>* index);

void deserialize(raft::resources const& handle,
                 const std::string& str,
                 cuvs::neighbors::ivf_flat::index<int8_t, int64_t>* index);

void deserialize(raft::resources const& handle,
                 const std::string& str,
                 cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>* index);
/**
 * @}
 */
}  // namespace cuvs::neighbors::ivf_flat