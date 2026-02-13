/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/common.hpp>

#include <raft/util/integer_utils.hpp>

#ifdef __cpp_lib_bitops
#include <bit>
#endif

namespace cuvs::neighbors {

/**
 * @brief Common VPQ dataset implementation - provides shared implementations.
 *
 * This class contains the common implementations for derived properties
 * that are shared between owning and view implementations.
 *
 * @tparam MathT the type of elements in the codebooks
 * @tparam IdxT type of the vector indices
 */
template <typename MathT, typename IdxT>
class vpq_dataset_impl : public vpq_dataset_iface<MathT, IdxT> {
 public:
  using index_type = IdxT;
  using math_type  = MathT;

  // Derived properties with default implementations
  [[nodiscard]] auto n_rows() const noexcept -> index_type override { return this->data().extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t override { return this->vq_code_book().extent(1); }
  [[nodiscard]] auto is_owning() const noexcept -> bool override { return true; }

  /** Row length of the encoded data in bytes. */
  [[nodiscard]] inline auto encoded_row_length() const noexcept -> uint32_t override
  {
    return this->data().extent(1);
  }
  /** The number of "coarse cluster centers" */
  [[nodiscard]] inline auto vq_n_centers() const noexcept -> uint32_t override
  {
    return this->vq_code_book().extent(0);
  }
  /** The bit length of an encoded vector element after compression by PQ. */
  [[nodiscard]] inline auto pq_bits() const noexcept -> uint32_t override
  {
    /*
    NOTE: pq_bits and the book size

    Normally, we'd store `pq_bits` as a part of the index.
    However, we know there's an invariant `pq_n_centers = 1 << pq_bits`, i.e. the codebook size is
    the same as the number of possible code values. Hence, we don't store the pq_bits and derive it
    from the array dimensions instead.
     */
    auto pq_width = pq_n_centers();
#ifdef __cpp_lib_bitops
    return std::countr_zero(pq_width);
#else
    uint32_t pq_bits = 0;
    while (pq_width > 1) {
      pq_bits++;
      pq_width >>= 1;
    }
    return pq_bits;
#endif
  }
  /** The dimensionality of an encoded vector after compression by PQ. */
  [[nodiscard]] inline auto pq_dim() const noexcept -> uint32_t override
  {
    return raft::div_rounding_up_unsafe(dim(), pq_len());
  }
  /** Dimensionality of a subspaces, i.e. the number of vector components mapped to a subspace */
  [[nodiscard]] inline auto pq_len() const noexcept -> uint32_t override
  {
    return this->pq_code_book().extent(1);
  }
  /** The number of vectors in a PQ codebook (`1 << pq_bits`). */
  [[nodiscard]] inline auto pq_n_centers() const noexcept -> uint32_t override
  {
    return this->pq_code_book().extent(0);
  }
};

/**
 * @brief Owning VPQ dataset implementation - owns the codebooks and data.
 *
 * @tparam MathT the type of elements in the codebooks
 * @tparam IdxT type of the vector indices
 */
template <typename MathT, typename IdxT>
class vpq_dataset_owning : public vpq_dataset_impl<MathT, IdxT> {
  public:
   using index_type = IdxT;
   using math_type  = MathT;

  /**
   * @brief Construct an owning vpq_dataset by moving in the codebooks and data.
   */
  vpq_dataset_owning(raft::device_matrix<math_type, uint32_t, raft::row_major>&& vq_code_book,
                     raft::device_matrix<math_type, uint32_t, raft::row_major>&& pq_code_book,
                     raft::device_matrix<uint8_t, index_type, raft::row_major>&& data)
    : vq_code_book_{std::move(vq_code_book)},
      pq_code_book_{std::move(pq_code_book)},
      data_{std::move(data)}
  {
  }

  vpq_dataset_owning(const vpq_dataset_owning&)            = delete;
  vpq_dataset_owning& operator=(const vpq_dataset_owning&) = delete;
  vpq_dataset_owning(vpq_dataset_owning&&)                 = default;
  vpq_dataset_owning& operator=(vpq_dataset_owning&&)      = default;
  ~vpq_dataset_owning() override                           = default;

  [[nodiscard]] auto vq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major> override
  {
    return vq_code_book_.view();
  }

  [[nodiscard]] auto pq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major> override
  {
    return pq_code_book_.view();
  }

  [[nodiscard]] auto data() const noexcept
    -> raft::device_matrix_view<const uint8_t, index_type, raft::row_major> override
  {
    return data_.view();
  }

 private:
  raft::device_matrix<math_type, uint32_t, raft::row_major> vq_code_book_;
  raft::device_matrix<math_type, uint32_t, raft::row_major> pq_code_book_;
  raft::device_matrix<uint8_t, index_type, raft::row_major> data_;
};

/**
 * @brief View-type VPQ dataset implementation - non-owning views to external data.
 *
 * The caller must ensure the lifetime of the underlying data exceeds
 * the lifetime of this object.
 *
 * @tparam MathT the type of elements in the codebooks
 * @tparam IdxT type of the vector indices
 */
template <typename MathT, typename IdxT>
class vpq_dataset_view : public vpq_dataset_impl<MathT, IdxT> {
  public:
  using index_type = IdxT;
  using math_type  = MathT;

  /**
   * @brief Construct a view-type vpq_dataset from external codebook views.
   *
   * @param vq_code_book_view View of VQ codebook [vq_n_centers, dim]
   * @param pq_code_book_view View of PQ codebook [pq_dim * pq_n_centers, pq_len] or [pq_n_centers,
   * pq_len]
   * @param data_view View of compressed data (can be empty for quantizer-only use)
   */
  vpq_dataset_view(
    raft::device_matrix_view<const math_type, uint32_t, raft::row_major> vq_code_book_view,
    raft::device_matrix_view<const math_type, uint32_t, raft::row_major> pq_code_book_view,
    raft::device_matrix_view<const uint8_t, index_type, raft::row_major> data_view =
      raft::device_matrix_view<const uint8_t, index_type, raft::row_major>{})
    : vq_code_book_view_{vq_code_book_view},
      pq_code_book_view_{pq_code_book_view},
      data_view_{data_view}
  {
  }

  vpq_dataset_view(const vpq_dataset_view&)            = default;
  vpq_dataset_view& operator=(const vpq_dataset_view&) = default;
  vpq_dataset_view(vpq_dataset_view&&)                 = default;
  vpq_dataset_view& operator=(vpq_dataset_view&&)      = default;
  ~vpq_dataset_view() override                         = default;

  [[nodiscard]] auto vq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major> override
  {
    return vq_code_book_view_;
  }

  [[nodiscard]] auto pq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major> override
  {
    return pq_code_book_view_;
  }

  [[nodiscard]] auto data() const noexcept
    -> raft::device_matrix_view<const uint8_t, index_type, raft::row_major> override
  {
    return data_view_;
  }

 private:
  raft::device_matrix_view<const math_type, uint32_t, raft::row_major> vq_code_book_view_;
  raft::device_matrix_view<const math_type, uint32_t, raft::row_major> pq_code_book_view_;
  raft::device_matrix_view<const uint8_t, index_type, raft::row_major> data_view_;
};

}  // namespace cuvs::neighbors
