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

#include <cuda_fp16.h>

#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/integer_utils.hpp>

namespace cuvs::neighbors::ivf_pq {

/**
 * @defgroup ivf_pq_cpp_index_params IVF-PQ index build parameters
 * @{
 */
/** A type for specifying how PQ codebooks are created. */
enum class codebook_gen {  // NOLINT
  PER_SUBSPACE = 0,        // NOLINT
  PER_CLUSTER  = 1,        // NOLINT
};

struct index_params : cuvs::neighbors::index_params {
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
  /**
   * Whether to add the dataset content to the index, i.e.:
   *
   *  - `true` means the index is filled with the dataset vectors and ready to search after calling
   * `build`.
   *  - `false` means `build` only trains the underlying model (e.g. quantizer or clustering), but
   * the index is left empty; you'd need to call `extend` on the index afterwards to populate it.
   */
  bool add_data_on_build = true;

  /**
   * The max number of data points to use per PQ code during PQ codebook training. Using more data
   * points per PQ code may increase the quality of PQ codebook but may also increase the build
   * time. The parameter is applied to both PQ codebook generation methods, i.e., PER_SUBSPACE and
   * PER_CLUSTER. In both cases, we will use `pq_book_size * max_train_points_per_pq_code` training
   * points to train each codebook.
   */
  uint32_t max_train_points_per_pq_code = 256;

  /**
   * Creates index_params based on shape of the input dataset.
   * Usage example:
   * @code{.cpp}
   *   using namespace cuvs::neighbors;
   *   raft::resources res;
   *   // create index_params for a [N. D] dataset and have InnerProduct as the distance metric
   *   auto dataset = raft::make_device_matrix<float, int64_t>(res, N, D);
   *   ivf_pq::index_params index_params =
   *     ivf_pq::index_params::from_dataset(dataset.extents(), raft::distance::InnerProduct);
   *   // modify/update index_params as needed
   *   index_params.add_data_on_build = true;
   * @endcode
   */
  static index_params from_dataset(
    raft::matrix_extent<int64_t> dataset,
    cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);
};
/**
 * @}
 */

/**
 * @defgroup ivf_pq_cpp_search_params IVF-PQ index search parameters
 * @{
 */
struct search_params : cuvs::neighbors::search_params {
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
};
/**
 * @}
 */

static_assert(std::is_aggregate_v<index_params>);
static_assert(std::is_aggregate_v<search_params>);

/** Size of the interleaved group. */
constexpr static uint32_t kIndexGroupSize = 32;
/** Stride of the interleaved group for vectorized loads. */
constexpr static uint32_t kIndexGroupVecLen = 16;

/**
 * Default value returned by `search` when the `n_probes` is too small and top-k is too large.
 * One may encounter it if the combined size of probed clusters is smaller than the requested
 * number of results per query.
 */
template <typename IdxT>
constexpr static IdxT kOutOfBoundsRecord = std::numeric_limits<IdxT>::max();

template <typename SizeT, typename IdxT>
struct list_spec {
  using value_type = uint8_t;
  using index_type = IdxT;
  /** PQ-encoded data stored in the interleaved format:
   *
   *    [ ceildiv(list_size, kIndexGroupSize)
   *    , ceildiv(pq_dim, (kIndexGroupVecLen * 8u) / pq_bits)
   *    , kIndexGroupSize
   *    , kIndexGroupVecLen
   *    ].
   */
  using list_extents = raft::
    extents<SizeT, raft::dynamic_extent, raft::dynamic_extent, kIndexGroupSize, kIndexGroupVecLen>;

  SizeT align_max;
  SizeT align_min;
  uint32_t pq_bits;
  uint32_t pq_dim;

  constexpr list_spec(uint32_t pq_bits, uint32_t pq_dim, bool conservative_memory_allocation);

  // Allow casting between different size-types (for safer size and offset calculations)
  template <typename OtherSizeT>
  constexpr explicit list_spec(const list_spec<OtherSizeT, IdxT>& other_spec);

  /** Determine the extents of an array enough to hold a given amount of data. */
  constexpr list_extents make_list_extents(SizeT n_rows) const;
};

template <typename SizeT, typename IdxT>
constexpr list_spec<SizeT, IdxT>::list_spec(uint32_t pq_bits,
                                            uint32_t pq_dim,
                                            bool conservative_memory_allocation)
  : pq_bits(pq_bits),
    pq_dim(pq_dim),
    align_min(kIndexGroupSize),
    align_max(conservative_memory_allocation ? kIndexGroupSize : 1024)
{
}

template <typename SizeT, typename IdxT>
template <typename OtherSizeT>
constexpr list_spec<SizeT, IdxT>::list_spec(const list_spec<OtherSizeT, IdxT>& other_spec)
  : pq_bits{other_spec.pq_bits},
    pq_dim{other_spec.pq_dim},
    align_min{other_spec.align_min},
    align_max{other_spec.align_max}
{
}

template <typename SizeT, typename IdxT>
constexpr typename list_spec<SizeT, IdxT>::list_extents list_spec<SizeT, IdxT>::make_list_extents(
  SizeT n_rows) const
{
  // how many elems of pq_dim fit into one kIndexGroupVecLen-byte chunk
  auto pq_chunk = (kIndexGroupVecLen * 8u) / pq_bits;
  return raft::make_extents<SizeT>(raft::div_rounding_up_safe<SizeT>(n_rows, kIndexGroupSize),
                                   raft::div_rounding_up_safe<SizeT>(pq_dim, pq_chunk),
                                   kIndexGroupSize,
                                   kIndexGroupVecLen);
}

template <typename IdxT, typename SizeT = uint32_t>
using list_data = ivf::list<list_spec, SizeT, IdxT>;

/**
 * @defgroup ivf_pq_cpp_index IVF-PQ index
 * @{
 */
/**
 * @brief IVF-PQ index.
 *
 * In the IVF-PQ index, a database vector y is approximated with two level quantization:
 *
 * y = Q_1(y) + Q_2(y - Q_1(y))
 *
 * The first level quantizer (Q_1), maps the vector y to the nearest cluster center. The number of
 * clusters is n_lists.
 *
 * The second quantizer encodes the residual, and it is defined as a product quantizer [1].
 *
 * A product quantizer encodes a `dim` dimensional vector with a `pq_dim` dimensional vector.
 * First we split the input vector into `pq_dim` subvectors (denoted by u), where each u vector
 * contains `pq_len` distinct components of y
 *
 * y_1, y_2, ... y_{pq_len}, y_{pq_len+1}, ... y_{2*pq_len}, ... y_{dim-pq_len+1} ... y_{dim}
 *  \___________________/     \____________________________/      \______________________/
 *         u_1                         u_2                          u_{pq_dim}
 *
 * Then each subvector encoded with a separate quantizer q_i, end the results are concatenated
 *
 * Q_2(y) = q_1(u_1),q_2(u_2),...,q_{pq_dim}(u_pq_dim})
 *
 * Each quantizer q_i outputs a code with pq_bit bits. The second level quantizers are also defined
 * by k-means clustering in the corresponding sub-space: the reproduction values are the centroids,
 * and the set of reproduction values is the codebook.
 *
 * When the data dimensionality `dim` is not multiple of `pq_dim`, the feature space is transformed
 * using a random orthogonal matrix to have `rot_dim = pq_dim * pq_len` dimensions
 * (`rot_dim >= dim`).
 *
 * The second-level quantizers are trained either for each subspace or for each cluster:
 *   (a) codebook_gen::PER_SUBSPACE:
 *         creates `pq_dim` second-level quantizers - one for each slice of the data along features;
 *   (b) codebook_gen::PER_CLUSTER:
 *         creates `n_lists` second-level quantizers - one for each first-level cluster.
 * In either case, the centroids are again found using k-means clustering interpreting the data as
 * having pq_len dimensions.
 *
 * [1] Product quantization for nearest neighbor search Herve Jegou, Matthijs Douze, Cordelia Schmid
 *
 * @tparam IdxT type of the indices in the source dataset
 *
 */
template <typename IdxT>
struct index : cuvs::neighbors::index {
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

  using pq_centers_extents = std::experimental::
    extents<uint32_t, raft::dynamic_extent, raft::dynamic_extent, raft::dynamic_extent>;

 public:
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

  /**
   * @brief Construct an empty index.
   *
   * Constructs an empty index. This index will either need to be trained with `build`
   * or loaded from a saved copy with `deserialize`
   */
  index(raft::resources const& handle);

  /** Construct an empty index. It needs to be trained and then populated. */
  index(raft::resources const& handle,
        cuvs::distance::DistanceType metric,
        codebook_gen codebook_kind,
        uint32_t n_lists,
        uint32_t dim,
        uint32_t pq_bits                    = 8,
        uint32_t pq_dim                     = 0,
        bool conservative_memory_allocation = false);

  /** Construct an empty index. It needs to be trained and then populated. */
  index(raft::resources const& handle, const index_params& params, uint32_t dim);

  /** Total length of the index. */
  IdxT size() const noexcept;

  /** Dimensionality of the input data. */
  uint32_t dim() const noexcept;

  /**
   * Dimensionality of the cluster centers:
   * input data dim extended with vector norms and padded to 8 elems.
   */
  uint32_t dim_ext() const noexcept;

  /**
   * Dimensionality of the data after transforming it for PQ processing
   * (rotated and augmented to be muplitple of `pq_dim`).
   */
  uint32_t rot_dim() const noexcept;

  /** The bit length of an encoded vector element after compression by PQ. */
  uint32_t pq_bits() const noexcept;

  /** The dimensionality of an encoded vector after compression by PQ. */
  uint32_t pq_dim() const noexcept;

  /** Dimensionality of a subspaces, i.e. the number of vector components mapped to a subspace */
  uint32_t pq_len() const noexcept;

  /** The number of vectors in a PQ codebook (`1 << pq_bits`). */
  uint32_t pq_book_size() const noexcept;

  /** Distance metric used for clustering. */
  cuvs::distance::DistanceType metric() const noexcept;

  /** How PQ codebooks are created. */
  codebook_gen codebook_kind() const noexcept;

  /** Number of clusters/inverted lists (first level quantization). */
  uint32_t n_lists() const noexcept;

  /**
   * Whether to use convervative memory allocation when extending the list (cluster) data
   * (see index_params.conservative_memory_allocation).
   */
  bool conservative_memory_allocation() const noexcept;

  /**
   * PQ cluster centers
   *
   *   - codebook_gen::PER_SUBSPACE: [pq_dim , pq_len, pq_book_size]
   *   - codebook_gen::PER_CLUSTER:  [n_lists, pq_len, pq_book_size]
   */
  raft::device_mdspan<float, pq_centers_extents, raft::row_major> pq_centers() noexcept;
  raft::device_mdspan<const float, pq_centers_extents, raft::row_major> pq_centers() const noexcept;

  /** Lists' data and indices. */
  std::vector<std::shared_ptr<list_data<IdxT>>>& lists() noexcept;
  const std::vector<std::shared_ptr<list_data<IdxT>>>& lists() const noexcept;

  /** Pointers to the inverted lists (clusters) data  [n_lists]. */
  raft::device_vector_view<uint8_t*, uint32_t, raft::row_major> data_ptrs() noexcept;
  raft::device_vector_view<const uint8_t* const, uint32_t, raft::row_major> data_ptrs()
    const noexcept;

  /** Pointers to the inverted lists (clusters) indices  [n_lists]. */
  raft::device_vector_view<IdxT*, uint32_t, raft::row_major> inds_ptrs() noexcept;
  raft::device_vector_view<const IdxT* const, uint32_t, raft::row_major> inds_ptrs() const noexcept;

  /** The transform matrix (original space -> rotated padded space) [rot_dim, dim] */
  raft::device_matrix_view<float, uint32_t, raft::row_major> rotation_matrix() noexcept;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix() const noexcept;

  /**
   * Accumulated list sizes, sorted in descending order [n_lists + 1].
   * The last value contains the total length of the index.
   * The value at index zero is always zero.
   *
   * That is, the content of this span is as if the `list_sizes` was sorted and then accumulated.
   *
   * This span is used during search to estimate the maximum size of the workspace.
   */
  raft::host_vector_view<IdxT, uint32_t, raft::row_major> accum_sorted_sizes() noexcept;
  raft::host_vector_view<const IdxT, uint32_t, raft::row_major> accum_sorted_sizes() const noexcept;

  /** Sizes of the lists [n_lists]. */
  raft::device_vector_view<uint32_t, uint32_t, raft::row_major> list_sizes() noexcept;
  raft::device_vector_view<const uint32_t, uint32_t, raft::row_major> list_sizes() const noexcept;

  /** Cluster centers corresponding to the lists in the original space [n_lists, dim_ext] */
  raft::device_matrix_view<float, uint32_t, raft::row_major> centers() noexcept;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers() const noexcept;

  /** Cluster centers corresponding to the lists in the rotated space [n_lists, rot_dim] */
  raft::device_matrix_view<float, uint32_t, raft::row_major> centers_rot() noexcept;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot() const noexcept;

  /** fetch size of a particular IVF list in bytes using the list extents.
   * Usage example:
   * @code{.cpp}
   *   raft::resources res;
   *   // use default index params
   *   ivf_pq::index_params index_params;
   *   // extend the IVF lists while building the index
   *   index_params.add_data_on_build = true;
   *   // create and fill the index from a [N, D] dataset
   *   auto index = cuvs::neighbors::ivf_pq::build<int64_t>(res, index_params, dataset, N, D);
   *   // Fetch the size of the fourth list
   *   uint32_t size = index.get_list_size_in_bytes(3);
   * @endcode
   *
   * @param[in] label list ID
   */
  uint32_t get_list_size_in_bytes(uint32_t label);

 private:
  cuvs::distance::DistanceType metric_;
  codebook_gen codebook_kind_;
  uint32_t dim_;
  uint32_t pq_bits_;
  uint32_t pq_dim_;
  bool conservative_memory_allocation_;

  // Primary data members
  std::vector<std::shared_ptr<list_data<IdxT>>> lists_;
  raft::device_vector<uint32_t, uint32_t, raft::row_major> list_sizes_;
  raft::device_mdarray<float, pq_centers_extents, raft::row_major> pq_centers_;
  raft::device_matrix<float, uint32_t, raft::row_major> centers_;
  raft::device_matrix<float, uint32_t, raft::row_major> centers_rot_;
  raft::device_matrix<float, uint32_t, raft::row_major> rotation_matrix_;

  // Computed members for accelerating search.
  raft::device_vector<uint8_t*, uint32_t, raft::row_major> data_ptrs_;
  raft::device_vector<IdxT*, uint32_t, raft::row_major> inds_ptrs_;
  raft::host_vector<IdxT, uint32_t, raft::row_major> accum_sorted_sizes_;

  /** Throw an error if the index content is inconsistent. */
  void check_consistency();

  pq_centers_extents make_pq_centers_extents();

  static uint32_t calculate_pq_dim(uint32_t dim);
};
/**
 * @}
 */

/**
 * @defgroup ivf_pq_cpp_index_build IVF-PQ index build
 * @{
 */
/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device matrix view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-pq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   ivf_pq::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_pq::build(handle, index_params, dataset, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::device_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_pq::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device matrix view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-pq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::device_matrix_view<const half, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   ivf_pq::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_pq::build(handle, index_params, dataset, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::device_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_pq::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_pq::index<int64_t>* idx);
/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device matrix view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-pq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   ivf_pq::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_pq::build(handle, index_params, dataset, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::device_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_pq::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device matrix view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-pq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   ivf_pq::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_pq::build(handle, index_params, dataset, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::device_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_pq::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_pq::index<int64_t>* idx);
/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a host_matrix_view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-pq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   ivf_pq::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_pq::build(handle, index_params, dataset, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_pq::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a host_matrix_view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-pq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::host_matrix_view<const half, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   ivf_pq::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_pq::build(handle, index_params, dataset, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_pq::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::host_matrix_view<const half, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a host_matrix_view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-pq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   ivf_pq::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_pq::build(handle, index_params, dataset, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_pq::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a host_matrix_view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-pq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   ivf_pq::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_pq::build(handle, index_params, dataset, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_pq::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_pq::index<int64_t>* idx);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_cpp_index_extend IVF-PQ index extend
 * @{
 */
/**
 * @brief Extend the index with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_pq::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Extend the index with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_pq::extend(handle, new_vectors, no_op, &index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
void extend(raft::resources const& handle,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Extend the index with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_pq::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Extend the index with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_pq::extend(handle, new_vectors, no_op, &index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
void extend(raft::resources const& handle,
            raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_pq::index<int64_t>* idx);
/**
 * @brief Extend the index with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_pq::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Extend the index with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_pq::extend(handle, new_vectors, no_op, &index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
void extend(raft::resources const& handle,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Extend the index with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_pq::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Extend the index with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_pq::extend(handle, new_vectors, no_op, &index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
void extend(raft::resources const& handle,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Extend the index with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_pq::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a host matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a host vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
auto extend(raft::resources const& handle,
            raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Extend the index with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_pq::extend(handle, new_vectors, no_op, &index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a host matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a host vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
void extend(raft::resources const& handle,
            raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Extend the index with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_pq::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a host matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a host vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
auto extend(raft::resources const& handle,
            raft::host_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Extend the index with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_pq::extend(handle, new_vectors, no_op, &index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a host matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a host vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
void extend(raft::resources const& handle,
            raft::host_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Extend the index with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_pq::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a host matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a host vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
auto extend(raft::resources const& handle,
            raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Extend the index with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_pq::extend(handle, new_vectors, no_op, &index_empty);
 *
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a host matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a host vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
void extend(raft::resources const& handle,
            raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_pq::index<int64_t>* idx);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_cpp_index_search IVF-PQ index search
 * @{
 */
/**
 * @brief Search ANN using the constructed index.
 *
 * See the [ivf_pq::build](#ivf_pq::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`.
 * The exact size of the temporary buffer depends on multiple factors and is an implementation
 * detail. However, you can safely specify a small initial size for the memory pool, so that only a
 * few allocations happen to grow it during the first invocations of the `search`.
 *
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_pq::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   ivf_pq::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   ivf_pq::search(handle, search_params, index, queries3, out_inds3, out_dists3);
 *   ...
 * @endcode
 *
 * @param[in] handle
 * @param[in] search_params configure the search
 * @param[in] index ivf-pq constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_pq::search_params& search_params,
            cuvs::neighbors::ivf_pq::index<int64_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [ivf_pq::build](#ivf_pq::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`.
 * The exact size of the temporary buffer depends on multiple factors and is an implementation
 * detail. However, you can safely specify a small initial size for the memory pool, so that only a
 * few allocations happen to grow it during the first invocations of the `search`.
 *
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_pq::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   ivf_pq::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   ivf_pq::search(handle, search_params, index, queries3, out_inds3, out_dists3);
 *   ...
 * @endcode
 *
 * @param[in] handle
 * @param[in] search_params configure the search
 * @param[in] index ivf-pq constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_pq::search_params& search_params,
            cuvs::neighbors::ivf_pq::index<int64_t>& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [ivf_pq::build](#ivf_pq::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`.
 * The exact size of the temporary buffer depends on multiple factors and is an implementation
 * detail. However, you can safely specify a small initial size for the memory pool, so that only a
 * few allocations happen to grow it during the first invocations of the `search`.
 *
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_pq::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   ivf_pq::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   ivf_pq::search(handle, search_params, index, queries3, out_inds3, out_dists3);
 *   ...
 * @endcode
 *
 * @param[in] handle
 * @param[in] search_params configure the search
 * @param[in] index ivf-pq constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_pq::search_params& search_params,
            cuvs::neighbors::ivf_pq::index<int64_t>& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [ivf_pq::build](#ivf_pq::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`.
 * The exact size of the temporary buffer depends on multiple factors and is an implementation
 * detail. However, you can safely specify a small initial size for the memory pool, so that only a
 * few allocations happen to grow it during the first invocations of the `search`.
 *
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_pq::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   ivf_pq::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   ivf_pq::search(handle, search_params, index, queries3, out_inds3, out_dists3);
 *   ...
 * @endcode
 *
 * @param[in] handle
 * @param[in] search_params configure the search
 * @param[in] index ivf-pq constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_pq::search_params& search_params,
            cuvs::neighbors::ivf_pq::index<int64_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/**
 * @brief Search ANN using the constructed index with the given filter.
 *
 * See the [ivf_pq::build](#ivf_pq::build) documentation for a usage example.
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
 * @param[in] idx ivf-pq constructed index
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
  index<int64_t>& idx,
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> sample_filter);

/**
 * @brief Search ANN using the constructed index with the given filter.
 *
 * See the [ivf_pq::build](#ivf_pq::build) documentation for a usage example.
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
 * @param[in] idx ivf-pq constructed index
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
  index<int64_t>& idx,
  raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> sample_filter);

/**
 * @brief Search ANN using the constructed index with the given filter.
 *
 * See the [ivf_pq::build](#ivf_pq::build) documentation for a usage example.
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
 * @param[in] idx ivf-pq constructed index
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
  index<int64_t>& idx,
  raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> sample_filter);

/**
 * @brief Search ANN using the constructed index with the given filter.
 *
 * See the [ivf_pq::build](#ivf_pq::build) documentation for a usage example.
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
 * @param[in] idx ivf-pq constructed index
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
  index<int64_t>& idx,
  raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> sample_filter);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_cpp_serialize IVF-PQ index serialize
 * @{
 */
/**
 * Write the index to an output stream
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = ivf_pq::build(...);`
 * cuvs::neighbors::ivf_pq::serialize(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index IVF-PQ index
 *
 */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::ivf_pq::index<int64_t>& index);

/**
 * Save the index to file.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = ivf_pq::build(...);`
 * cuvs::neighbors::ivf_pq::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-PQ index
 *
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::ivf_pq::index<int64_t>& index);

/**
 * Load index from input stream
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 *
 * using IdxT = int64_t; // type of the index
 * // create an empty index
 * cuvs::neighbors::ivf_pq::index<IdxT> index(handle);
 *
 * cuvs::neighbors::ivf_pq::deserialize(handle, is, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] str the name of the file that stores the index
 * @param[out] index IVF-PQ index
 *
 */
void deserialize(raft::resources const& handle,
                 std::istream& str,
                 cuvs::neighbors::ivf_pq::index<int64_t>* index);
/**
 * Load index from file.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * using IdxT = int64_t; // type of the index
 * // create an empty index
 * ivf_pq::index<IdxT> index(handle);
 *
 * cuvs::neighbors::ivf_pq::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index IVF-PQ index
 *
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_pq::index<int64_t>* index);
/**
 * @}
 */

namespace helpers {
/**
 * @defgroup ivf_pq_cpp_helpers IVF-PQ helper methods
 * @{
 */
namespace codepacker {
/**
 * @addtogroup ivf_pq_cpp_helpers
 * @{
 */
/**
 * @brief Unpack `n_take` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`.
 *
 * Bit compression is removed, which means output will have pq_dim dimensional vectors (one code per
 * byte, instead of ceildiv(pq_dim * pq_bits, 8) bytes of pq codes).
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data = index.lists()[label]->data.view();
 *   // allocate the buffer for the output
 *   uint32_t n_take = 4;
 *   auto codes = raft::make_device_matrix<uint8_t>(res, n_take, index.pq_dim());
 *   uint32_t offset = 0;
 *   // unpack n_take elements from the list
 *   ivf_pq::helpers::codepacker::unpack(res, list_data, index.pq_bits(), offset, codes.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] list_data block to read from
 * @param[in] pq_bits bit length of encoded vector elements
 * @param[in] offset
 *   How many records in the list to skip.
 * @param[out] codes
 *   the destination buffer [n_take, index.pq_dim()].
 *   The length `n_take` defines how many records to unpack,
 *   it must be smaller than the list size.
 */
void unpack(
  raft::resources const& res,
  raft::device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data,
  uint32_t pq_bits,
  uint32_t offset,
  raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> codes);

/**
 * @brief Unpack `n_rows` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`. The output codes of a single vector are contiguous, not expanded to
 * one code per byte, which means the output has ceildiv(pq_dim * pq_bits, 8) bytes per PQ encoded
 * vector.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   auto list_data = index.lists()[label]->data.view();
 *   // allocate the buffer for the output
 *   uint32_t n_rows = 4;
 *   auto codes = raft::make_device_matrix<uint8_t>(
 *     res, n_rows, raft::ceildiv(index.pq_dim() * index.pq_bits(), 8));
 *   uint32_t offset = 0;
 *   // unpack n_rows elements from the list
 *   ivf_pq::helpers::codepacker::unpack_contiguous(
 *     res, list_data, index.pq_bits(), offset, n_rows, index.pq_dim(), codes.data_handle());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] list_data block to read from
 * @param[in] pq_bits bit length of encoded vector elements
 * @param[in] offset
 *   How many records in the list to skip.
 * @param[in] n_rows How many records to unpack
 * @param[in] pq_dim The dimensionality of the PQ compressed records
 * @param[out] codes
 *   the destination buffer [n_rows, ceildiv(pq_dim * pq_bits, 8)].
 *   The length `n_rows` defines how many records to unpack,
 *   it must be smaller than the list size.
 */
void unpack_contiguous(
  raft::resources const& res,
  raft::device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data,
  uint32_t pq_bits,
  uint32_t offset,
  uint32_t n_rows,
  uint32_t pq_dim,
  uint8_t* codes);

/**
 * Write flat PQ codes into an existing list by the given offset.
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_vec).
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data  = index.lists()[label]->data.view();
 *   // allocate the buffer for the input codes
 *   auto codes = raft::make_device_matrix<uint8_t>(res, n_vec, index.pq_dim());
 *   ... prepare n_vecs to pack into the list in codes ...
 *   // write codes into the list starting from the 42nd position
 *   ivf_pq::helpers::codepacker::pack(
 *       res, make_const_mdspan(codes.view()), index.pq_bits(), 42, list_data);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] codes flat PQ codes, one code per byte [n_vec, pq_dim]
 * @param[in] pq_bits bit length of encoded vector elements
 * @param[in] offset how many records to skip before writing the data into the list
 * @param[in] list_data block to write into
 */
void pack(raft::resources const& res,
          raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
          uint32_t pq_bits,
          uint32_t offset,
          raft::device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
            list_data);

/**
 * Write flat PQ codes into an existing list by the given offset. The input codes of a single vector
 * are contiguous (not expanded to one code per byte).
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_rows records).
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   auto list_data  = index.lists()[label]->data.view();
 *   // allocate the buffer for the input codes
 *   auto codes = raft::make_device_matrix<uint8_t>(
 *     res, n_rows, raft::ceildiv(index.pq_dim() * index.pq_bits(), 8));
 *   ... prepare compressed vectors to pack into the list in codes ...
 *   // write codes into the list starting from the 42nd position. If the current size of the list
 *   // is greater than 42, this will overwrite the codes starting at this offset.
 *   ivf_pq::helpers::codepacker::pack_contiguous(
 *     res, codes.data_handle(), n_rows, index.pq_dim(), index.pq_bits(), 42, list_data);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] codes flat PQ codes, [n_vec, ceildiv(pq_dim * pq_bits, 8)]
 * @param[in] n_rows number of records
 * @param[in] pq_dim
 * @param[in] pq_bits bit length of encoded vector elements
 * @param[in] offset how many records to skip before writing the data into the list
 * @param[in] list_data block to write into
 */
void pack_contiguous(
  raft::resources const& res,
  const uint8_t* codes,
  uint32_t n_rows,
  uint32_t pq_dim,
  uint32_t pq_bits,
  uint32_t offset,
  raft::device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data);

/**
 * Write flat PQ codes into an existing list by the given offset.
 *
 * The list is identified by its label.
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_vec).
 *
 * Usage example:
 * @code{.cpp}
 *   // We will write into the 137th cluster
 *   uint32_t label = 137;
 *   // allocate the buffer for the input codes
 *   auto codes = raft::make_device_matrix<const uint8_t>(res, n_vec, index.pq_dim());
 *   ... prepare n_vecs to pack into the list in codes ...
 *   // write codes into the list starting from the 42nd position
 *   ivf_pq::helpers::codepacker::pack_list_data(res, &index, codes_to_pack, label, 42);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index IVF-PQ index.
 * @param[in] codes flat PQ codes, one code per byte [n_rows, pq_dim]
 * @param[in] label The id of the list (cluster) into which we write.
 * @param[in] offset how many records to skip before writing the data into the list
 */
void pack_list_data(raft::resources const& res,
                    index<int64_t>* index,
                    raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
                    uint32_t label,
                    uint32_t offset);

/**
 * Write flat PQ codes into an existing list by the given offset. Use this when the input
 * vectors are PQ encoded and not expanded to one code per byte.
 *
 * The list is identified by its label.
 *
 * NB: no memory allocation happens here; the list into which the vectors are packed must fit offset
 * + n_rows rows.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   raft::resources res;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(res, index_params, dataset, N, D);
 *   // allocate the buffer for n_rows input codes. Each vector occupies
 *   // raft::ceildiv(index.pq_dim() * index.pq_bits(), 8) bytes because
 *   // codes are compressed and without gaps.
 *   auto codes = raft::make_device_matrix<const uint8_t>(
 *     res, n_rows, raft::ceildiv(index.pq_dim() * index.pq_bits(), 8));
 *   ... prepare the compressed vectors to pack into the list in codes ...
 *   // the first n_rows codes in the fourth IVF list are to be overwritten.
 *   uint32_t label = 3;
 *   // write codes into the list starting from the 0th position
 *   ivf_pq::helpers::codepacker::pack_contiguous_list_data(
 *     res, &index, codes.data_handle(), n_rows, label, 0);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-PQ index
 * @param[in] codes flat contiguous PQ codes [n_rows, ceildiv(pq_dim * pq_bits, 8)]
 * @param[in] n_rows how many records to pack
 * @param[in] label The id of the list (cluster) into which we write.
 * @param[in] offset how many records to skip before writing the data into the list
 */
void pack_contiguous_list_data(raft::resources const& res,
                               index<int64_t>* index,
                               uint8_t* codes,
                               uint32_t n_rows,
                               uint32_t label,
                               uint32_t offset);

/**
 * @brief Unpack `n_take` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`, one code per byte (independently of pq_bits).
 *
 * Usage example:
 * @code{.cpp}
 *   // We will unpack the fourth cluster
 *   uint32_t label = 3;
 *   // Get the list size
 *   uint32_t list_size = 0;
 *   raft::copy(&list_size, index.list_sizes().data_handle() + label, 1,
 * resource::get_cuda_stream(res)); resource::sync_stream(res);
 *   // allocate the buffer for the output
 *   auto codes = raft::make_device_matrix<float>(res, list_size, index.pq_dim());
 *   // unpack the whole list
 *   ivf_pq::helpers::codepacker::unpack_list_data(res, index, codes.view(), label, 0);
 * @endcode
 *
 * @param[in] res
 * @param[in] index
 * @param[out] out_codes
 *   the destination buffer [n_take, index.pq_dim()].
 *   The length `n_take` defines how many records to unpack,
 *   it must be smaller than the list size.
 * @param[in] label
 *   The id of the list (cluster) to decode.
 * @param[in] offset
 *   How many records in the list to skip.
 */
void unpack_list_data(raft::resources const& res,
                      const index<int64_t>& index,
                      raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes,
                      uint32_t label,
                      uint32_t offset);

/**
 * @brief Unpack a series of records of a single list (cluster) in the compressed index
 * by their in-list offsets, one code per byte (independently of pq_bits).
 *
 * Usage example:
 * @code{.cpp}
 *   // We will unpack the fourth cluster
 *   uint32_t label = 3;
 *   // Create the selection vector
 *   auto selected_indices = raft::make_device_vector<uint32_t>(res, 4);
 *   ... fill the indices ...
 *   resource::sync_stream(res);
 *   // allocate the buffer for the output
 *   auto codes = raft::make_device_matrix<float>(res, selected_indices.size(), index.pq_dim());
 *   // decode the whole list
 *   ivf_pq::helpers::codepacker::unpack_list_data(
 *       res, index, selected_indices.view(), codes.view(), label);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] index IVF-PQ index (passed by reference)
 * @param[in] in_cluster_indices
 *   The offsets of the selected indices within the cluster.
 * @param[out] out_codes
 *   the destination buffer [n_take, index.pq_dim()].
 *   The length `n_take` defines how many records to unpack,
 *   it must be smaller than the list size.
 * @param[in] label
 *   The id of the list (cluster) to decode.
 */
void unpack_list_data(raft::resources const& res,
                      const index<int64_t>& index,
                      raft::device_vector_view<const uint32_t> in_cluster_indices,
                      raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes,
                      uint32_t label);

/**
 * @brief Unpack `n_rows` consecutive PQ encoded vectors of a single list (cluster) in the
 * compressed index starting at given `offset`, not expanded to one code per byte. Each code in the
 * output buffer occupies ceildiv(index.pq_dim() * index.pq_bits(), 8) bytes.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   // We will unpack the whole fourth cluster
 *   uint32_t label = 3;
 *   // Get the list size
 *   uint32_t list_size = 0;
 *   raft::update_host(&list_size, index.list_sizes().data_handle() + label, 1,
 *     raft::resource::get_cuda_stream(res));
 *   raft::resource::sync_stream(res);
 *   // allocate the buffer for the output
 *   auto codes = raft::make_device_matrix<float>(res, list_size, raft::ceildiv(index.pq_dim() *
 *     index.pq_bits(), 8));
 *   // unpack the whole list
 *   ivf_pq::helpers::codepacker::unpack_list_data(res, index, codes.data_handle(), list_size,
 * label, 0);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] index IVF-PQ index (passed by reference)
 * @param[out] out_codes
 *   the destination buffer [n_rows, ceildiv(index.pq_dim() * index.pq_bits(), 8)].
 *   The length `n_rows` defines how many records to unpack,
 *   offset + n_rows must be smaller than or equal to the list size.
 * @param[in] n_rows how many codes to unpack
 * @param[in] label
 *   The id of the list (cluster) to decode.
 * @param[in] offset
 *   How many records in the list to skip.
 */
void unpack_contiguous_list_data(raft::resources const& res,
                                 const index<int64_t>& index,
                                 uint8_t* out_codes,
                                 uint32_t n_rows,
                                 uint32_t label,
                                 uint32_t offset);

/**
 * @brief Decode `n_take` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`.
 *
 * Usage example:
 * @code{.cpp}
 *   // We will reconstruct the fourth cluster
 *   uint32_t label = 3;
 *   // Get the list size
 *   uint32_t list_size = 0;
 *   raft::copy(&list_size, index.list_sizes().data_handle() + label, 1,
 *   resource::get_cuda_stream(res)); resource::sync_stream(res);
 *   // allocate the buffer for the output
 *   auto decoded_vectors = raft::make_device_matrix<float>(res, list_size, index.dim());
 *   // decode the whole list
 *   ivf_pq::helpers::codepacker::reconstruct_list_data(res, index, decoded_vectors.view(), label,
 * 0);
 * @endcode
 *
 * @param[in] res
 * @param[in] index
 * @param[out] out_vectors
 *   the destination buffer [n_take, index.dim()].
 *   The length `n_take` defines how many records to reconstruct,
 *   it must be smaller than the list size.
 * @param[in] label
 *   The id of the list (cluster) to decode.
 * @param[in] offset
 *   How many records in the list to skip.
 */
void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_matrix_view<float, uint32_t, raft::row_major> out_vectors,
                           uint32_t label,
                           uint32_t offset);

void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_matrix_view<half, uint32_t, raft::row_major> out_vectors,
                           uint32_t label,
                           uint32_t offset);

void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_matrix_view<int8_t, uint32_t, raft::row_major> out_vectors,
                           uint32_t label,
                           uint32_t offset);

void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_vectors,
                           uint32_t label,
                           uint32_t offset);

/**
 * @brief Decode a series of records of a single list (cluster) in the compressed index
 * by their in-list offsets.
 *
 * Usage example:
 * @code{.cpp}
 *   // We will reconstruct the fourth cluster
 *   uint32_t label = 3;
 *   // Create the selection vector
 *   auto selected_indices = raft::make_device_vector<uint32_t>(res, 4);
 *   ... fill the indices ...
 *   resource::sync_stream(res);
 *   // allocate the buffer for the output
 *   auto decoded_vectors = raft::make_device_matrix<float>(
 *                             res, selected_indices.size(), index.dim());
 *   // decode the whole list
 *   ivf_pq::helpers::codepacker::reconstruct_list_data(
 *       res, index, selected_indices.view(), decoded_vectors.view(), label);
 * @endcode
 *
 * @param[in] res
 * @param[in] index
 * @param[in] in_cluster_indices
 *   The offsets of the selected indices within the cluster.
 * @param[out] out_vectors
 *   the destination buffer [n_take, index.dim()].
 *   The length `n_take` defines how many records to reconstruct,
 *   it must be smaller than the list size.
 * @param[in] label
 *   The id of the list (cluster) to decode.
 */
void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_vector_view<const uint32_t> in_cluster_indices,
                           raft::device_matrix_view<float, uint32_t, raft::row_major> out_vectors,
                           uint32_t label);
void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_vector_view<const uint32_t> in_cluster_indices,
                           raft::device_matrix_view<half, uint32_t, raft::row_major> out_vectors,
                           uint32_t label);
void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_vector_view<const uint32_t> in_cluster_indices,
                           raft::device_matrix_view<int8_t, uint32_t, raft::row_major> out_vectors,
                           uint32_t label);
void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_vector_view<const uint32_t> in_cluster_indices,
                           raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_vectors,
                           uint32_t label);

/**
 * @brief Extend one list of the index in-place, by the list label, skipping the classification and
 * encoding steps.
 *
 * Usage example:
 * @code{.cpp}
 *   // We will extend the fourth cluster
 *   uint32_t label = 3;
 *   // We will fill 4 new vectors
 *   uint32_t n_vec = 4;
 *   // Indices of the new vectors
 *   auto indices = raft::make_device_vector<uint32_t>(res, n_vec);
 *   ... fill the indices ...
 *   auto new_codes = raft::make_device_matrix<uint8_t, uint32_t, row_major> new_codes(
 *       res, n_vec, index.pq_dim());
 *   ... fill codes ...
 *   // extend list with new codes
 *   ivf_pq::helpers::codepacker::extend_list_with_codes(
 *       res, &index, codes.view(), indices.view(), label);
 * @endcode
 *
 * @param[in] res
 * @param[inout] index
 * @param[in] new_codes flat PQ codes, one code per byte [n_rows, index.pq_dim()]
 * @param[in] new_indices source indices [n_rows]
 * @param[in] label the id of the target list (cluster).
 */
void extend_list_with_codes(
  raft::resources const& res,
  index<int64_t>* index,
  raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> new_codes,
  raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
  uint32_t label);

/**
 * @brief Extend one list of the index in-place, by the list label, skipping the classification
 * step.
 *
 *  Usage example:
 * @code{.cpp}
 *   // We will extend the fourth cluster
 *   uint32_t label = 3;
 *   // We will extend with 4 new vectors
 *   uint32_t n_vec = 4;
 *   // Indices of the new vectors
 *   auto indices = raft::make_device_vector<uint32_t>(res, n_vec);
 *   ... fill the indices ...
 *   auto new_vectors = raft::make_device_matrix<float, uint32_t, row_major> new_codes(
 *       res, n_vec, index.dim());
 *   ... fill vectors ...
 *   // extend list with new vectors
 *   ivf_pq::helpers::codepacker::extend_list(
 *       res, &index, new_vectors.view(), indices.view(), label);
 * @endcode
 *
 *
 * @param[in] res
 * @param[inout] index
 * @param[in] new_vectors data to encode [n_rows, index.dim()]
 * @param[in] new_indices source indices [n_rows]
 * @param[in] label the id of the target list (cluster).
 */
void extend_list(raft::resources const& res,
                 index<int64_t>* index,
                 raft::device_matrix_view<const float, uint32_t, raft::row_major> new_vectors,
                 raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
                 uint32_t label);
void extend_list(raft::resources const& res,
                 index<int64_t>* index,
                 raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> new_vectors,
                 raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
                 uint32_t label);
void extend_list(raft::resources const& res,
                 index<int64_t>* index,
                 raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> new_vectors,
                 raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
                 uint32_t label);

/**
 * @}
 */
};  // namespace codepacker

/**
 * @brief Remove all data from a single list (cluster) in the index.
 *
 * Usage example:
 * @code{.cpp}
 *   // We will erase the fourth cluster (label = 3)
 *   ivf_pq::helpers::erase_list(res, &index, 3);
 * @endcode
 *
 *
 * @param[in] res
 * @param[inout] index
 * @param[in] label the id of the target list (cluster).
 */
void erase_list(raft::resources const& res, index<int64_t>* index, uint32_t label);

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
 *   ivf_pq::index_params index_params;
 *   // initialize an empty index
 *   ivf_pq::index<int64_t> index(res, index_params, D);
 *   // reset the index's state and list sizes
 *   ivf_pq::helpers::reset_index(res, &index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-PQ index
 */
void reset_index(const raft::resources& res, index<int64_t>* index);

/**
 * @brief Public helper API exposing the computation of the index's rotation matrix.
 * NB: This is to be used only when the rotation matrix is not already computed through
 * cuvs::neighbors::ivf_pq::build.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // force random rotation
 *   index_params.force_random_rotation = true;
 *   // initialize an empty index
 *   cuvs::neighbors::ivf_pq::index<int64_t> index(res, index_params, D);
 *   // reset the index
 *   reset_index(res, &index);
 *   // compute the rotation matrix with random_rotation
 *   cuvs::neighbors::ivf_pq::helpers::make_rotation_matrix(
 *     res, &index, index_params.force_random_rotation);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-PQ index
 * @param[in] force_random_rotation whether to apply a random rotation matrix on the input data. See
 * cuvs::neighbors::ivf_pq::index_params for more details.
 */
void make_rotation_matrix(raft::resources const& res,
                          index<int64_t>* index,
                          bool force_random_rotation);

/**
 * @brief Public helper API for externally modifying the index's IVF centroids.
 * NB: The index must be reset before this. Use raft::neighbors::ivf_pq::extend to construct IVF
 lists according to new centroids.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   // allocate the buffer for the input centers
 *   auto cluster_centers = raft::make_device_matrix<float, uint32_t>(res, index.n_lists(),
 index.dim());
 *   ... prepare ivf centroids in cluster_centers ...
 *   // reset the index
 *   reset_index(res, &index);
 *   // recompute the state of the index
 *   cuvs::neighbors::ivf_pq::helpers::recompute_internal_state(res, index);
 *   // Write the IVF centroids
 *   cuvs::neighbors::ivf_pq::helpers::set_centers(
                    res,
                    &index,
                    cluster_centers);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-PQ index
 * @param[in] cluster_centers new cluster centers [index.n_lists(), index.dim()]
 */
void set_centers(raft::resources const& res,
                 index<int64_t>* index,
                 raft::device_matrix_view<const float, uint32_t> cluster_centers);

/**
 * @brief Public helper API for fetching a trained index's IVF centroids
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   // allocate the buffer for the output centers
 *   auto cluster_centers = raft::make_device_matrix<float, uint32_t>(
 *     res, index.n_lists(), index.dim());
 *   // Extract the IVF centroids into the buffer
 *   cuvs::neighbors::ivf_pq::helpers::extract_centers(res, index, cluster_centers.data_handle());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] index IVF-PQ index (passed by reference)
 * @param[out] cluster_centers IVF cluster centers [index.n_lists(), index.dim]
 */
void extract_centers(raft::resources const& res,
                     const index<int64_t>& index,
                     raft::device_matrix_view<float, uint32_t, raft::row_major> cluster_centers);

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
void recompute_internal_state(const raft::resources& res, index<int64_t>* index);

/**
 * @}
 */
}  // namespace helpers

}  // namespace cuvs::neighbors::ivf_pq
