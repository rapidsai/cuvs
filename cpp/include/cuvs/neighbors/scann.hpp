/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/stream_view.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/integer_utils.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <optional>
#include <variant>

namespace cuvs::neighbors::experimental::scann {
/**
 * @defgroup scann_cpp_index_params ScaNN index build parameters
 * @{
 */

/**
 * @brief ANN parameters used by ScaNN to build index
 *
 */
struct index_params : cuvs::neighbors::index_params {
  // partitioning parameters

  /** the number of leaves in the tree **/
  uint32_t n_leaves = 1000;
  /** the number of rows for training the tree structures **/
  int64_t kmeans_n_rows_train = 100000;

  /** the max number of iterations for training the tree structure **/
  uint32_t kmeans_n_iters = 24;

  /** the value of eta for AVQ adjustment during partitioning **/
  float partitioning_eta = 1;

  /** the value of lambda for SOAR spilling **/
  float soar_lambda = 1;

  // Residual quanitzation params
  /** the dimension of pq subspaces (must divide dataset dimension)**/
  uint32_t pq_dim = 8;

  /** the number of bits for pq codes (must be 4 or 8, for 16 and 256 codes respectively) **/
  uint32_t pq_bits = 8;

  /** the number of rows for PQ training (internally capped to 100k) **/
  int64_t pq_n_rows_train = 100000;

  /** the max number of iterations for PQ training **/
  uint32_t pq_train_iters = 10;

  /** whether to apply bf16 quantization of dataset vectors **/
  bool bf16_enabled = false;

  // TODO - add other scann build params
};

/**
 * @}
 */

static_assert(std::is_aggregate_v<index_params>);

/**
 * @defgroup scann_cpp_index ScaNN index type
 * @{
 */

/**
 * @brief ScaNN index.
 *
 * The index stores the dataset and the ScaNN graph in device memory.
 *
 * @tparam T data element type
 * @tparam IdxT type of the vector indices (represent dataset.extent(0))
 *
 */
template <typename T, typename IdxT>
struct index : cuvs::neighbors::index {
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

 public:
  /** Distance metric used for clustering. */
  [[nodiscard]] constexpr inline auto metric() const noexcept -> cuvs::distance::DistanceType
  {
    return metric_;
  }

  /** Total length of the index (number of vectors). */
  IdxT size() const noexcept;

  /** Dimensionality of the data. */
  [[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t { return dim_; }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

  /** Construct an empty index. It will need to be trained and populated with `build`*/
  //  index(raft::resources const& res) {}

  index(raft::resources const& res,
        cuvs::distance::DistanceType metric,
        uint32_t n_leaves,
        uint32_t pq_bits,
        uint32_t pq_dim,
        IdxT n_rows,
        IdxT dim,
        uint32_t pq_clusters,
        uint32_t pq_num_subspaces,
        bool bf16_enabled)
    : cuvs::neighbors::index(),
      metric_(metric),
      pq_dim_(pq_dim),
      pq_bits_(pq_bits),
      n_leaves_(n_leaves),
      centers_(raft::make_device_matrix<float, IdxT>(res, n_leaves, dim)),
      labels_(raft::make_device_vector<uint32_t, IdxT>(res, n_rows)),
      soar_labels_(raft::make_device_vector<uint32_t, IdxT>(res, n_rows)),
      pq_codebook_(
        raft::make_device_matrix<float, uint32_t, raft::row_major>(res, pq_clusters, dim)),
      quantized_residuals_(
        raft::make_host_matrix<uint8_t, IdxT, raft::row_major>(n_rows, pq_num_subspaces)),
      quantized_soar_residuals_(
        raft::make_host_matrix<uint8_t, IdxT, raft::row_major>(n_rows, pq_num_subspaces)),
      n_rows_(n_rows),
      dim_(dim),
      bf16_dataset_(raft::make_host_matrix<int16_t, IdxT, raft::row_major>(
        bf16_enabled ? n_rows : 0, bf16_enabled ? dim : 0))

  {
  }

  index(raft::resources const& res, const index_params& params, IdxT n_rows, IdxT dim)
    : index(res,
            params.metric,
            params.n_leaves,
            params.pq_bits,
            params.pq_dim,
            n_rows,
            dim,
            1 << params.pq_bits,
            dim / params.pq_dim,
            params.bf16_enabled)
  {
    RAFT_EXPECTS(params.pq_bits == 4 || params.pq_bits == 8, "ScaNN only supports 4 or 8 bit PQ");
    RAFT_EXPECTS(dim >= params.pq_dim,
                 "PQ subspace dimension (pq_dim) should be smaller than the dataset dimension");
    RAFT_EXPECTS(dim % params.pq_dim == 0,
                 "PQ subspace dimension (pq_dim) must divide the dataset dimension");
  }

  raft::device_matrix_view<float, IdxT> centers() noexcept { return centers_.view(); }

  raft::device_matrix_view<const float, IdxT> centers() const noexcept
  {
    return raft::make_const_mdspan(centers_.view());
  }

  raft::device_vector_view<uint32_t, IdxT> labels() noexcept { return labels_.view(); }

  raft::device_vector_view<const uint32_t, IdxT> labels() const noexcept
  {
    return raft::make_const_mdspan(labels_.view());
  }

  raft::device_vector_view<uint32_t, IdxT> soar_labels() noexcept { return soar_labels_.view(); }

  raft::device_vector_view<const uint32_t, IdxT> soar_labels() const noexcept
  {
    return raft::make_const_mdspan(soar_labels_.view());
  }

  uint32_t n_rows() const noexcept { return n_rows_; }

  uint32_t n_leaves() const noexcept { return n_leaves_; }

  uint32_t pq_dim() const noexcept { return pq_dim_; }

  raft::device_matrix_view<const float, uint32_t, raft::row_major> pq_codebook() const noexcept
  {
    return raft::make_const_mdspan(pq_codebook_.view());
  }

  raft::device_matrix_view<float, uint32_t, raft::row_major> pq_codebook() noexcept
  {
    return pq_codebook_.view();
  }

  raft::host_matrix_view<const uint8_t, IdxT, raft::row_major> quantized_residuals() const noexcept
  {
    return raft::make_const_mdspan(quantized_residuals_.view());
  }

  raft::host_matrix_view<uint8_t, IdxT, raft::row_major> quantized_residuals() noexcept
  {
    return quantized_residuals_.view();
  }

  raft::host_matrix_view<const uint8_t, IdxT, raft::row_major> quantized_soar_residuals()
    const noexcept
  {
    return raft::make_const_mdspan(quantized_soar_residuals_.view());
  }

  raft::host_matrix_view<uint8_t, IdxT, raft::row_major> quantized_soar_residuals() noexcept
  {
    return quantized_soar_residuals_.view();
  }

  raft::host_matrix_view<int16_t, IdxT, raft::row_major> bf16_dataset() noexcept
  {
    return bf16_dataset_.view();
  }

  raft::host_matrix_view<const int16_t, IdxT, raft::row_major> bf16_dataset() const noexcept
  {
    return raft::make_const_mdspan(bf16_dataset_.view());
  }

 private:
  cuvs::distance::DistanceType metric_;
  IdxT dim_;
  IdxT n_rows_;
  uint32_t pq_dim_;
  uint32_t pq_bits_;
  uint32_t n_leaves_;

  raft::device_matrix<float, IdxT, raft::row_major> centers_;
  raft::device_vector<uint32_t, IdxT> labels_;
  raft::device_vector<uint32_t, IdxT> soar_labels_;
  raft::device_matrix<float, uint32_t, raft::row_major> pq_codebook_;
  raft::host_matrix<uint8_t, IdxT, raft::row_major> quantized_residuals_;
  raft::host_matrix<uint8_t, IdxT, raft::row_major> quantized_soar_residuals_;
  raft::host_matrix<int16_t, IdxT, raft::row_major> bf16_dataset_;
  // TODO - add any data, pointers or structures needed
};
/**
 * @}
 */

/**
 * @defgroup scann_cpp_index_build ScaNN index build functions
 * @{
 */
/**
 * @brief Build the index from the dataset for efficient search.
 *
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::experimental::scann::index_params& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::experimental::scann::index<float, int64_t>;

auto build(raft::resources const& handle,
           const cuvs::neighbors::experimental::scann::index_params& params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::experimental::scann::index<float, int64_t>;
/**
 * @defgroup scann_cpp_serialize ScaNN serialize functions
 * @{
 */
/**
 * @brief Save the index to files in a directory
 *
 * This serializes the index into a list of files for integration into
 * OSS ScaNN for use with search
 *
 * NOTE: the implementation of ScaNN index build is EXPERIMENTAL and currently
 * not subject to comprehensive, automated testing. Accuracy and performance
 * are not guaranteed, and could diverge without warning.
 *
 */

void serialize(raft::resources const& handle,
               const std::string& file_prefix,
               const cuvs::neighbors::experimental::scann::index<float, int64_t>& index);

/**
 * @}
 */

}  // namespace cuvs::neighbors::experimental::scann
