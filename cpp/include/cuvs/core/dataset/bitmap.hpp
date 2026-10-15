/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>

#include <raft/core/bitmap.hpp>
#include <raft/core/bitset.hpp>
#include <raft/core/error.hpp>

#include <cstdint>
#include <type_traits>

namespace CUVS_EXPORT cuvs {

/**
 * @brief A non-owning view of a row-major bitmap dataset.
 *
 * The rows are packed without padding: bit `(row, col)` is stored at linear bit offset
 * `row * dim() + col`. The referenced storage must outlive the view and every operation using it.
 *
 * @tparam BitmapT Underlying bitmap word type (`uint32_t` or `uint64_t`).
 * @tparam IdxT Row index type.
 */
template <typename BitmapT, typename IdxT>
struct device_bitmap_dataset_view {
  static_assert(std::is_same_v<BitmapT, uint32_t> || std::is_same_v<BitmapT, uint64_t>,
                "BitmapT must be uint32_t or uint64_t.");

  using value_type           = bool;
  using index_type           = IdxT;
  using code_type            = BitmapT;
  using const_data_view_type = raft::core::bitmap_view<const BitmapT, IdxT>;

  device_bitmap_dataset_view() = default;

  /**
   * @brief Construct a bitmap dataset view.
   *
   * @param data Pointer to tightly packed device bitmap words.
   * @param n_rows Number of dataset rows.
   * @param dim Logical number of bits in each row.
   */
  device_bitmap_dataset_view(const BitmapT* data, IdxT n_rows, IdxT dim) : data_{data, n_rows, dim}
  {
  }

  /** @return Number of dataset rows. */
  [[nodiscard]] auto n_rows() const noexcept -> IdxT { return data_.get_n_rows(); }

  /** @return Logical number of bits in each row. */
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(data_.get_n_cols());
  }

  /** @return Read-only view of the tightly packed bitmap data. */
  [[nodiscard]] auto data_const_view() const noexcept -> const_data_view_type { return data_; }

 private:
  const_data_view_type data_{nullptr, 0, 0};
};

/**
 * @brief Make a bitmap dataset view over an owning RAFT bitset.
 *
 * @param bitset Owning bitset containing exactly `n_rows * dim` logical bits.
 * @param n_rows Number of dataset rows.
 * @param dim Logical number of bits in each row.
 */
template <typename BitmapT, typename IdxT>
[[nodiscard]] auto make_device_bitmap_dataset_view(const raft::core::bitset<BitmapT, IdxT>& bitset,
                                                   IdxT n_rows,
                                                   IdxT dim)
  -> device_bitmap_dataset_view<BitmapT, IdxT>
{
  RAFT_EXPECTS(n_rows >= 0 && dim >= 0, "Bitmap dataset extents must be non-negative.");
  RAFT_EXPECTS(bitset.size() == n_rows * dim, "Bitmap storage size must equal n_rows * dim.");
  return {bitset.data(), n_rows, dim};
}

/**
 * @brief Make a bitmap dataset view from a RAFT bitmap view.
 *
 * @param bitmap Existing bitmap view.
 */
template <typename BitmapT, typename IdxT>
[[nodiscard]] auto make_device_bitmap_dataset_view(
  const raft::core::bitmap_view<BitmapT, IdxT>& bitmap)
  -> device_bitmap_dataset_view<std::remove_const_t<BitmapT>, IdxT>
{
  using word_type = std::remove_const_t<BitmapT>;
  return device_bitmap_dataset_view<word_type, IdxT>{
    bitmap.data(), bitmap.get_n_rows(), bitmap.get_n_cols()};
}

}  // namespace CUVS_EXPORT cuvs
