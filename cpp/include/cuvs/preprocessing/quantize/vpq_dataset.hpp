/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/common.hpp>

namespace cuvs::preprocessing::quantize::pq {

/**
 * @brief VPQ compressed dataset - internal interface.
 *
 * This is the abstract base class for the internal implementation.
 * Users should use vpq_dataset which wraps this via PIMPL.
 *
 * @tparam MathT the type of elements in the codebooks
 * @tparam IdxT type of the vector indices (represent dataset.extent(0))
 */
template <typename MathT, typename IdxT>
class vpq_dataset_iface : public cuvs::neighbors::dataset<IdxT> {
 public:
  using index_type = IdxT;
  using math_type  = MathT;

  ~vpq_dataset_iface() override = default;

  /** Get view of VQ codebook. */
  [[nodiscard]] virtual auto vq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major> = 0;

  /** Get view of PQ codebook. */
  [[nodiscard]] virtual auto pq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major> = 0;

  /** Get view of compressed data. */
  [[nodiscard]] virtual auto data() const noexcept
    -> raft::device_matrix_view<const uint8_t, index_type, raft::row_major> = 0;

  // Derived properties - pure virtual
  [[nodiscard]] virtual auto n_rows() const noexcept -> index_type           = 0;
  [[nodiscard]] virtual auto dim() const noexcept -> uint32_t                = 0;
  [[nodiscard]] virtual auto encoded_row_length() const noexcept -> uint32_t = 0;
  [[nodiscard]] virtual auto vq_n_centers() const noexcept -> uint32_t       = 0;
  [[nodiscard]] virtual auto pq_bits() const noexcept -> uint32_t            = 0;
  [[nodiscard]] virtual auto pq_dim() const noexcept -> uint32_t             = 0;
  [[nodiscard]] virtual auto pq_len() const noexcept -> uint32_t             = 0;
  [[nodiscard]] virtual auto pq_n_centers() const noexcept -> uint32_t       = 0;
};

/**
 * @addtogroup pq
 * @{
 */

/**
 * @brief VPQ compressed dataset (PIMPL wrapper).
 *
 * The dataset is compressed using two level quantization:
 *   1. Vector Quantization
 *   2. Product Quantization of residuals
 *
 * This class wraps the internal implementation (vpq_dataset_owning or vpq_dataset_view)
 * and provides a stable API.
 *
 * @tparam MathT the type of elements in the codebooks
 * @tparam IdxT type of the vector indices (represent dataset.extent(0))
 */
template <typename MathT, typename IdxT>
class vpq_dataset : public cuvs::neighbors::dataset<IdxT> {
 public:
  using index_type = IdxT;
  using math_type  = MathT;

  vpq_dataset() = default;

  /** Construct from an implementation. */
  explicit vpq_dataset(std::unique_ptr<vpq_dataset_iface<MathT, IdxT>> impl)
    : impl_{std::move(impl)}
  {
  }

  vpq_dataset(const vpq_dataset&)            = delete;
  vpq_dataset& operator=(const vpq_dataset&) = delete;
  vpq_dataset(vpq_dataset&&)                 = default;
  vpq_dataset& operator=(vpq_dataset&&)      = default;
  ~vpq_dataset() override                    = default;

  [[nodiscard]] auto n_rows() const noexcept -> index_type override { return impl_->n_rows(); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t override { return impl_->dim(); }
  [[nodiscard]] auto is_owning() const noexcept -> bool final { return true; }

  /** Get view of VQ codebook. */
  [[nodiscard]] auto vq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major>
  {
    return impl_->vq_code_book();
  }

  /** Get view of PQ codebook. */
  [[nodiscard]] auto pq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major>
  {
    return impl_->pq_code_book();
  }

  /** Get view of compressed data. */
  [[nodiscard]] auto data() const noexcept
    -> raft::device_matrix_view<const uint8_t, index_type, raft::row_major>
  {
    return impl_->data();
  }

  /** Row length of the encoded data in bytes. */
  [[nodiscard]] auto encoded_row_length() const noexcept -> uint32_t
  {
    return impl_->encoded_row_length();
  }

  /** The number of "coarse cluster centers" */
  [[nodiscard]] auto vq_n_centers() const noexcept -> uint32_t { return impl_->vq_n_centers(); }

  /** The bit length of an encoded vector element after compression by PQ. */
  [[nodiscard]] auto pq_bits() const noexcept -> uint32_t { return impl_->pq_bits(); }

  /** The dimensionality of an encoded vector after compression by PQ. */
  [[nodiscard]] auto pq_dim() const noexcept -> uint32_t { return impl_->pq_dim(); }

  /** Dimensionality of a subspaces, i.e. the number of vector components mapped to a subspace */
  [[nodiscard]] auto pq_len() const noexcept -> uint32_t { return impl_->pq_len(); }

  /** The number of vectors in a PQ codebook (`1 << pq_bits`). */
  [[nodiscard]] auto pq_n_centers() const noexcept -> uint32_t { return impl_->pq_n_centers(); }

 private:
  std::unique_ptr<vpq_dataset_iface<MathT, IdxT>> impl_;
};

template <typename DatasetT>
struct is_vpq_dataset : std::false_type {};

template <typename MathT, typename IdxT>
struct is_vpq_dataset<vpq_dataset<MathT, IdxT>> : std::true_type {};

template <typename DatasetT>
inline constexpr bool is_vpq_dataset_v = is_vpq_dataset<DatasetT>::value;

/** @} */  // end of group pq

}  // namespace cuvs::preprocessing::quantize::pq
