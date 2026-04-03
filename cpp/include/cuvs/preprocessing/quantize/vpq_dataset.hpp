/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/common.hpp>

namespace cuvs::preprocessing::quantize::pq {

// ============================================================================
// vpq_codebooks — VQ + PQ codebook storage (owning or view)
// ============================================================================

/**
 * @brief Abstract interface for VPQ codebook access.
 *
 * @tparam MathT the type of elements in the codebooks
 */
template <typename MathT>
class vpq_codebooks_iface {
 public:
  using math_type = MathT;

  virtual ~vpq_codebooks_iface() = default;

  /** Get view of VQ codebook [vq_n_centers, dim]. */
  [[nodiscard]] virtual auto vq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major> = 0;

  /** Get view of PQ codebook [pq_n_centers (× pq_dim for subspaces), pq_len]. */
  [[nodiscard]] virtual auto pq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major> = 0;

  [[nodiscard]] virtual auto dim() const noexcept -> uint32_t          = 0;
  [[nodiscard]] virtual auto vq_n_centers() const noexcept -> uint32_t = 0;
  [[nodiscard]] virtual auto pq_bits() const noexcept -> uint32_t      = 0;
  [[nodiscard]] virtual auto pq_dim() const noexcept -> uint32_t       = 0;
  [[nodiscard]] virtual auto pq_len() const noexcept -> uint32_t       = 0;
  [[nodiscard]] virtual auto pq_n_centers() const noexcept -> uint32_t = 0;
};

/**
 * @addtogroup pq
 * @{
 */

/**
 * @brief PIMPL wrapper for VQ + PQ codebooks.
 *
 * Internally delegates to either an owning implementation (holds device
 * matrices) or a view implementation (references external device memory).
 *
 * @tparam MathT the type of elements in the codebooks
 */
template <typename MathT>
class vpq_codebooks {
 public:
  using math_type = MathT;

  vpq_codebooks() = default;

  /** Construct from an implementation. */
  explicit vpq_codebooks(std::unique_ptr<vpq_codebooks_iface<MathT>> impl) : impl_{std::move(impl)}
  {
  }

  vpq_codebooks(const vpq_codebooks&)            = delete;
  vpq_codebooks& operator=(const vpq_codebooks&) = delete;
  vpq_codebooks(vpq_codebooks&&)                 = default;
  vpq_codebooks& operator=(vpq_codebooks&&)      = default;
  ~vpq_codebooks()                               = default;

  [[nodiscard]] auto vq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major>
  {
    return impl_->vq_code_book();
  }

  [[nodiscard]] auto pq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major>
  {
    return impl_->pq_code_book();
  }

  [[nodiscard]] auto dim() const noexcept -> uint32_t { return impl_->dim(); }
  [[nodiscard]] auto vq_n_centers() const noexcept -> uint32_t { return impl_->vq_n_centers(); }
  [[nodiscard]] auto pq_bits() const noexcept -> uint32_t { return impl_->pq_bits(); }
  [[nodiscard]] auto pq_dim() const noexcept -> uint32_t { return impl_->pq_dim(); }
  [[nodiscard]] auto pq_len() const noexcept -> uint32_t { return impl_->pq_len(); }
  [[nodiscard]] auto pq_n_centers() const noexcept -> uint32_t { return impl_->pq_n_centers(); }

  /** Check whether this object has been initialised. */
  [[nodiscard]] explicit operator bool() const noexcept { return impl_ != nullptr; }

 private:
  std::unique_ptr<vpq_codebooks_iface<MathT>> impl_;
};

// ============================================================================
// vpq_dataset — codebooks + encoded data (always owning)
// ============================================================================

/**
 * @brief VPQ compressed dataset.
 *
 * Holds a set of VQ + PQ codebooks together with the encoded dataset.
 * Both the codebooks and the encoded data are always owned by this object.
 *
 * @tparam MathT the type of elements in the codebooks
 * @tparam IdxT  type of the vector indices (represent dataset.extent(0))
 */
template <typename MathT, typename IdxT>
class vpq_dataset : public cuvs::neighbors::dataset<IdxT> {
 public:
  using index_type = IdxT;
  using math_type  = MathT;

  vpq_dataset() = default;

  vpq_dataset(vpq_codebooks<MathT>&& codebooks,
              raft::device_matrix<uint8_t, IdxT, raft::row_major>&& data)
    : codebooks_{std::move(codebooks)}, data_{std::move(data)}
  {
  }

  vpq_dataset(const vpq_dataset&)            = delete;
  vpq_dataset& operator=(const vpq_dataset&) = delete;
  vpq_dataset(vpq_dataset&&)                 = default;
  vpq_dataset& operator=(vpq_dataset&&)      = default;
  ~vpq_dataset() override                    = default;

  // ── dataset<IdxT> interface ──────────────────────────────────────────────
  [[nodiscard]] auto n_rows() const noexcept -> index_type override { return data_.extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t override { return codebooks_.dim(); }
  [[nodiscard]] auto is_owning() const noexcept -> bool override { return true; }

  // ── Codebook access (convenience forwards) ───────────────────────────────
  [[nodiscard]] auto vq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major>
  {
    return codebooks_.vq_code_book();
  }
  [[nodiscard]] auto pq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major>
  {
    return codebooks_.pq_code_book();
  }

  /** Get view of the encoded (compressed) data. */
  [[nodiscard]] auto data() const noexcept
    -> raft::device_matrix_view<const uint8_t, index_type, raft::row_major>
  {
    return data_.view();
  }

  // ── Derived properties ───────────────────────────────────────────────────
  [[nodiscard]] auto encoded_row_length() const noexcept -> uint32_t { return data_.extent(1); }
  [[nodiscard]] auto vq_n_centers() const noexcept -> uint32_t { return codebooks_.vq_n_centers(); }
  [[nodiscard]] auto pq_bits() const noexcept -> uint32_t { return codebooks_.pq_bits(); }
  [[nodiscard]] auto pq_dim() const noexcept -> uint32_t { return codebooks_.pq_dim(); }
  [[nodiscard]] auto pq_len() const noexcept -> uint32_t { return codebooks_.pq_len(); }
  [[nodiscard]] auto pq_n_centers() const noexcept -> uint32_t { return codebooks_.pq_n_centers(); }

  /** Direct access to the codebooks object. */
  [[nodiscard]] auto codebooks() const noexcept -> const vpq_codebooks<MathT>&
  {
    return codebooks_;
  }

 private:
  vpq_codebooks<MathT> codebooks_;
  raft::device_matrix<uint8_t, index_type, raft::row_major> data_;
};

// ── Type trait ─────────────────────────────────────────────────────────────

template <typename DatasetT>
struct is_vpq_dataset : std::false_type {};

template <typename MathT, typename IdxT>
struct is_vpq_dataset<vpq_dataset<MathT, IdxT>> : std::true_type {};

template <typename DatasetT>
inline constexpr bool is_vpq_dataset_v = is_vpq_dataset<DatasetT>::value;

/** @} */  // end of group pq

}  // namespace cuvs::preprocessing::quantize::pq
