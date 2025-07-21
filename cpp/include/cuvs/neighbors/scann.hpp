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
  // partitioning
  uint32_t scann_clusters;
  int64_t kmeans_n_rows_train;
  uint32_t kmeans_n_iters = 24;
  float partitioning_eta  = 1;
  float soar_lambda;

  // Residual quanitzation params
  uint32_t pq_dim  = 8;
  uint32_t pq_bits = 8;
  int64_t pq_n_rows_train;
  uint32_t pq_train_iters = 10;

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
        uint32_t scann_clusters,
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
      scann_clusters_(scann_clusters),
      centers_(raft::make_device_matrix<float, IdxT>(res, scann_clusters, dim)),
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
            params.scann_clusters,
            params.pq_bits,
            params.pq_dim,
            n_rows,
            dim,
            1 << params.pq_bits,
            dim / params.pq_dim,
            params.bf16_enabled)
  {
    // consider addind checks on parameters (.e.g <= 8 bits pq)
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

  uint32_t n_clusters() const noexcept { return scann_clusters_; }

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
  uint32_t scann_clusters_;

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

auto build(raft::resources const& handle,
           const cuvs::neighbors::experimental::scann::index_params& params,
           raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::experimental::scann::index<int8_t, int64_t>;

auto build(raft::resources const& handle,
           const cuvs::neighbors::experimental::scann::index_params& params,
           raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::experimental::scann::index<int8_t, int64_t>;

auto build(raft::resources const& handle,
           const cuvs::neighbors::experimental::scann::index_params& params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::experimental::scann::index<uint8_t, int64_t>;

auto build(raft::resources const& handle,
           const cuvs::neighbors::experimental::scann::index_params& params,
           raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::experimental::scann::index<uint8_t, int64_t>;

/**
 * @defgroup scann_cpp_serialize ScaNN serialize functions
 * @{
 */
/**
 * Save the index to file.
 */

void serialize(raft::resources const& handle,
               const std::string& file_prefix,
               const cuvs::neighbors::experimental::scann::index<float, int64_t>& index);

void serialize(raft::resources const& handle,
               const std::string& file_prefix,
               const cuvs::neighbors::experimental::scann::index<int8_t, int64_t>& index);

void serialize(raft::resources const& handle,
               const std::string& file_prefix,
               const cuvs::neighbors::experimental::scann::index<uint8_t, int64_t>& index);

/**
 * @}
 */

}  // namespace cuvs::neighbors::experimental::scann
