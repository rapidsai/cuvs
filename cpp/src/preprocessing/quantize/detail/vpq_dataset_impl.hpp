/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/preprocessing/quantize/vpq_dataset.hpp>

namespace cuvs::preprocessing::quantize::pq {

template <typename MathT>
class vpq_codebooks_owning : public vpq_codebooks_iface<MathT> {
 public:
  using math_type = MathT;

  vpq_codebooks_owning(raft::device_matrix<math_type, uint32_t, raft::row_major>&& vq_code_book,
                       raft::device_matrix<math_type, uint32_t, raft::row_major>&& pq_code_book)
    : vq_code_book_{std::move(vq_code_book)}, pq_code_book_{std::move(pq_code_book)}
  {
  }

  vpq_codebooks_owning(const vpq_codebooks_owning&)            = delete;
  vpq_codebooks_owning& operator=(const vpq_codebooks_owning&) = delete;
  vpq_codebooks_owning(vpq_codebooks_owning&&)                 = default;
  vpq_codebooks_owning& operator=(vpq_codebooks_owning&&)      = default;
  ~vpq_codebooks_owning() override                             = default;

  [[nodiscard]] auto vq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major> override
  {
    return vq_code_book_.view();
  }

  [[nodiscard]] auto vq_code_book() noexcept
    -> raft::device_matrix_view<math_type, uint32_t, raft::row_major>
  {
    return vq_code_book_.view();
  }

  [[nodiscard]] auto pq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major> override
  {
    return pq_code_book_.view();
  }

  [[nodiscard]] auto pq_code_book() noexcept
    -> raft::device_matrix_view<math_type, uint32_t, raft::row_major>
  {
    return pq_code_book_.view();
  }

 private:
  raft::device_matrix<math_type, uint32_t, raft::row_major> vq_code_book_;
  raft::device_matrix<math_type, uint32_t, raft::row_major> pq_code_book_;
};

template <typename MathT>
class vpq_codebooks_view : public vpq_codebooks_iface<MathT> {
 public:
  using math_type = MathT;

  vpq_codebooks_view(
    raft::device_matrix_view<const math_type, uint32_t, raft::row_major> vq_code_book_view,
    raft::device_matrix_view<const math_type, uint32_t, raft::row_major> pq_code_book_view)
    : vq_code_book_view_{vq_code_book_view}, pq_code_book_view_{pq_code_book_view}
  {
  }

  vpq_codebooks_view(const vpq_codebooks_view&)            = default;
  vpq_codebooks_view& operator=(const vpq_codebooks_view&) = default;
  vpq_codebooks_view(vpq_codebooks_view&&)                 = default;
  vpq_codebooks_view& operator=(vpq_codebooks_view&&)      = default;
  ~vpq_codebooks_view() override                           = default;

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

 private:
  raft::device_matrix_view<const math_type, uint32_t, raft::row_major> vq_code_book_view_;
  raft::device_matrix_view<const math_type, uint32_t, raft::row_major> pq_code_book_view_;
};

}  // namespace cuvs::preprocessing::quantize::pq
