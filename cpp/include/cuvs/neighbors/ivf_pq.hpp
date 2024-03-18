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
#include <raft/neighbors/ivf_pq_types.hpp>

namespace cuvs::neighbors::ivf_pq {

/** A type for specifying how PQ codebooks are created. */
enum class codebook_gen {  // NOLINT
  PER_SUBSPACE = 0,        // NOLINT
  PER_CLUSTER  = 1,        // NOLINT
};

struct index_params : ann::index_params {
  /**
   * The number of inverted lists (clusters)
   *
   * Hint: the number of vectors per cluster (`n_rows/n_lists`) should be approximately 1,000 to
   * 10,000.
   */
  uint32_t n_lists = 1024;
  /** The number of iterations searching for kmeans centers (index building). */
  uint32_t kmeans_n_iters = 20;
  /** The fraction of data to use during iterative kmeans building. */
  double kmeans_trainset_fraction = 0.5;
  /**
   * The bit length of the vector element after compression by PQ.
   *
   * Possible values: [4, 5, 6, 7, 8].
   *
   * Hint: the smaller the 'pq_bits', the smaller the index size and the better the search
   * performance, but the lower the recall.
   */
  uint32_t pq_bits = 8;
  /**
   * The dimensionality of the vector after compression by PQ. When zero, an optimal value is
   * selected using a heuristic.
   *
   * NB: `pq_dim * pq_bits` must be a multiple of 8.
   *
   * Hint: a smaller 'pq_dim' results in a smaller index size and better search performance, but
   * lower recall. If 'pq_bits' is 8, 'pq_dim' can be set to any number, but multiple of 8 are
   * desirable for good performance. If 'pq_bits' is not 8, 'pq_dim' should be a multiple of 8.
   * For good performance, it is desirable that 'pq_dim' is a multiple of 32. Ideally, 'pq_dim'
   * should be also a divisor of the dataset dim.
   */
  uint32_t pq_dim = 0;
  /** How PQ codebooks are created. */
  codebook_gen codebook_kind = codebook_gen::PER_SUBSPACE;
  /**
   * Apply a random rotation matrix on the input data and queries even if `dim % pq_dim == 0`.
   *
   * Note: if `dim` is not multiple of `pq_dim`, a random rotation is always applied to the input
   * data and queries to transform the working space from `dim` to `rot_dim`, which may be slightly
   * larger than the original space and and is a multiple of `pq_dim` (`rot_dim % pq_dim == 0`).
   * However, this transform is not necessary when `dim` is multiple of `pq_dim`
   *   (`dim == rot_dim`, hence no need in adding "extra" data columns / features).
   *
   * By default, if `dim == rot_dim`, the rotation transform is initialized with the identity
   * matrix. When `force_random_rotation == true`, a random orthogonal transform matrix is generated
   * regardless of the values of `dim` and `pq_dim`.
   */
  bool force_random_rotation = false;
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

  /** Build a raft IVF_PQ index params from an existing cuvs IVF_PQ index params. */
  operator raft::neighbors::ivf_pq::index_params() const
  {
    return raft::neighbors::ivf_pq::index_params{
      {
        .metric            = static_cast<raft::distance::DistanceType>((int)this->metric),
        .metric_arg        = this->metric_arg,
        .add_data_on_build = this->add_data_on_build,
      },
      .n_lists                  = n_lists,
      .kmeans_n_iters           = kmeans_n_iters,
      .kmeans_trainset_fraction = kmeans_trainset_fraction,
      .pq_bits                  = pq_bits,
      .pq_dim                   = pq_dim,
      .codebook_kind = static_cast<raft::neighbors::ivf_pq::codebook_gen>((int)this->codebook_kind),
      .force_random_rotation          = force_random_rotation,
      .conservative_memory_allocation = conservative_memory_allocation};
  }
};

struct search_params : ann::search_params {
  /** The number of clusters to search. */
  uint32_t n_probes = 20;
  /**
   * Data type of look up table to be created dynamically at search time.
   *
   * Possible values: [CUDA_R_32F, CUDA_R_16F, CUDA_R_8U]
   *
   * The use of low-precision types reduces the amount of shared memory required at search time, so
   * fast shared memory kernels can be used even for datasets with large dimansionality. Note that
   * the recall is slightly degraded when low-precision type is selected.
   */
  cudaDataType_t lut_dtype = CUDA_R_32F;
  /**
   * Storage data type for distance/similarity computed at search time.
   *
   * Possible values: [CUDA_R_16F, CUDA_R_32F]
   *
   * If the performance limiter at search time is device memory access, selecting FP16 will improve
   * performance slightly.
   */
  cudaDataType_t internal_distance_dtype = CUDA_R_32F;
  /**
   * Preferred fraction of SM's unified memory / L1 cache to be used as shared memory.
   *
   * Possible values: [0.0 - 1.0] as a fraction of the `sharedMemPerMultiprocessor`.
   *
   * One wants to increase the carveout to make sure a good GPU occupancy for the main search
   * kernel, but not to keep it too high to leave some memory to be used as L1 cache. Note, this
   * value is interpreted only as a hint. Moreover, a GPU usually allows only a fixed set of cache
   * configurations, so the provided value is rounded up to the nearest configuration. Refer to the
   * NVIDIA tuning guide for the target GPU architecture.
   *
   * Note, this is a low-level tuning parameter that can have drastic negative effects on the search
   * performance if tweaked incorrectly.
   */
  double preferred_shmem_carveout = 1.0;

  /** Build a raft IVF_PQ search params from an existing cuvs IVF_PQ search params. */
  operator raft::neighbors::ivf_pq::search_params() const
  {
    raft::neighbors::ivf_pq::search_params result = {
      {}, n_probes, lut_dtype, internal_distance_dtype, preferred_shmem_carveout};
    return result;
  }
};

template <typename IdxT, typename SizeT = uint32_t>
using list_data = raft::neighbors::ivf_pq::list_data<IdxT, SizeT>;

template <typename IdxT>
struct index : ann::index {
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

  using pq_centers_extents = typename raft::neighbors::ivf_pq::index<IdxT>::pq_centers_extents;

 public:
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;
  index(raft::resources const& handle, const index_params& params, uint32_t dim);
  index(raft::neighbors::ivf_pq::index<IdxT>&& raft_idx);

  IdxT size() const noexcept;
  uint32_t dim() const noexcept;
  uint32_t dim_ext() const noexcept;
  uint32_t rot_dim() const noexcept;
  uint32_t pq_bits() const noexcept;
  uint32_t pq_dim() const noexcept;
  uint32_t pq_len() const noexcept;
  uint32_t pq_book_size() const noexcept;
  cuvs::distance::DistanceType metric() const noexcept;
  codebook_gen codebook_kind() const noexcept;
  uint32_t n_lists() const noexcept;
  bool conservative_memory_allocation() const noexcept;
  raft::mdspan<float, pq_centers_extents, raft::row_major> pq_centers() noexcept;
  raft::mdspan<const float, pq_centers_extents, raft::row_major> pq_centers() const noexcept;
  std::vector<std::shared_ptr<list_data<IdxT>>>& lists() noexcept;
  const std::vector<std::shared_ptr<list_data<IdxT>>>& lists() const noexcept;
  raft::device_vector_view<uint8_t*, uint32_t, raft::row_major> data_ptrs() noexcept;
  raft::device_vector_view<const uint8_t* const, uint32_t, raft::row_major> data_ptrs()
    const noexcept;
  raft::device_vector_view<IdxT*, uint32_t, raft::row_major> inds_ptrs() noexcept;
  raft::device_vector_view<const IdxT* const, uint32_t, raft::row_major> inds_ptrs() const noexcept;
  raft::device_matrix_view<float, uint32_t, raft::row_major> rotation_matrix() noexcept;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix() const noexcept;
  raft::host_vector_view<IdxT, uint32_t, raft::row_major> accum_sorted_sizes() noexcept;
  raft::host_vector_view<const IdxT, uint32_t, raft::row_major> accum_sorted_sizes() const noexcept;
  raft::device_vector_view<uint32_t, uint32_t, raft::row_major> list_sizes() noexcept;
  raft::device_vector_view<const uint32_t, uint32_t, raft::row_major> list_sizes() const noexcept;
  raft::device_matrix_view<float, uint32_t, raft::row_major> centers() noexcept;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers() const noexcept;
  raft::device_matrix_view<float, uint32_t, raft::row_major> centers_rot() noexcept;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot() const noexcept;
  const raft::neighbors::ivf_pq::index<IdxT>* get_raft_index() const;
  raft::neighbors::ivf_pq::index<IdxT>* get_raft_index();

 private:
  std::unique_ptr<raft::neighbors::ivf_pq::index<IdxT>> raft_index_;
};

#define CUVS_IVF_PQ(T, IdxT)                                                         \
  auto build(raft::resources const& handle,                                          \
             const cuvs::neighbors::ivf_pq::index_params& params,                    \
             raft::device_matrix_view<const T, IdxT, raft::row_major> dataset)       \
    ->cuvs::neighbors::ivf_pq::index<IdxT>;                                          \
                                                                                     \
  void build(raft::resources const& handle,                                          \
             const cuvs::neighbors::ivf_pq::index_params& params,                    \
             raft::device_matrix_view<const T, IdxT, raft::row_major> dataset,       \
             cuvs::neighbors::ivf_pq::index<IdxT>* idx);                             \
                                                                                     \
  auto extend(raft::resources const& handle,                                         \
              raft::device_matrix_view<const T, IdxT, raft::row_major> new_vectors,  \
              std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices, \
              const cuvs::neighbors::ivf_pq::index<IdxT>& orig_index)                \
    ->cuvs::neighbors::ivf_pq::index<IdxT>;                                          \
                                                                                     \
  void extend(raft::resources const& handle,                                         \
              raft::device_matrix_view<const T, IdxT, raft::row_major> new_vectors,  \
              std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices, \
              cuvs::neighbors::ivf_pq::index<IdxT>* idx);                            \
                                                                                     \
  void search(raft::resources const& handle,                                         \
              const cuvs::neighbors::ivf_pq::search_params& params,                  \
              cuvs::neighbors::ivf_pq::index<IdxT>& index,                           \
              raft::device_matrix_view<const T, IdxT, raft::row_major> queries,      \
              raft::device_matrix_view<IdxT, IdxT, raft::row_major> neighbors,       \
              raft::device_matrix_view<float, IdxT, raft::row_major> distances);     \
                                                                                     \
  void serialize(raft::resources const& handle,                                      \
                 std::string& filename,                                              \
                 const cuvs::neighbors::ivf_pq::index<IdxT>& index);                 \
                                                                                     \
  void deserialize(raft::resources const& handle,                                    \
                   const std::string& filename,                                      \
                   cuvs::neighbors::ivf_pq::index<IdxT>* index);

CUVS_IVF_PQ(float, int64_t);
CUVS_IVF_PQ(int8_t, int64_t);
CUVS_IVF_PQ(uint8_t, int64_t);

#undef CUVS_IVF_PQ

}  // namespace cuvs::neighbors::ivf_pq
