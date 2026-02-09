/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/ivf_pq.hpp>

namespace cuvs::neighbors::ivf_pq {

template <typename IdxT>
class index_impl : public index_iface<IdxT> {
 public:
  index_impl(raft::resources const& handle,
             cuvs::distance::DistanceType metric,
             codebook_gen codebook_kind,
             uint32_t n_lists,
             uint32_t dim,
             uint32_t pq_bits,
             uint32_t pq_dim,
             bool conservative_memory_allocation,
             list_layout codes_layout = list_layout::INTERLEAVED);

  ~index_impl()                                    = default;
  index_impl(index_impl&&)                         = default;
  auto operator=(index_impl&&) -> index_impl&      = default;
  index_impl(const index_impl&)                    = delete;
  auto operator=(const index_impl&) -> index_impl& = delete;

  [[nodiscard]] auto metric() const noexcept -> cuvs::distance::DistanceType override;
  [[nodiscard]] auto codebook_kind() const noexcept -> codebook_gen override;
  [[nodiscard]] auto codes_layout() const noexcept -> list_layout override;
  auto size() const noexcept -> IdxT override;
  [[nodiscard]] auto dim() const noexcept -> uint32_t override;
  [[nodiscard]] auto dim_ext() const noexcept -> uint32_t override;
  [[nodiscard]] auto rot_dim() const noexcept -> uint32_t override;
  [[nodiscard]] auto pq_bits() const noexcept -> uint32_t override;
  [[nodiscard]] auto pq_dim() const noexcept -> uint32_t override;
  [[nodiscard]] auto pq_len() const noexcept -> uint32_t override;
  [[nodiscard]] auto pq_book_size() const noexcept -> uint32_t override;
  [[nodiscard]] auto n_lists() const noexcept -> uint32_t override;
  [[nodiscard]] auto conservative_memory_allocation() const noexcept -> bool override;

  auto lists() noexcept -> std::vector<std::shared_ptr<list_data_base<IdxT>>>& override;
  [[nodiscard]] auto lists() const noexcept
    -> const std::vector<std::shared_ptr<list_data_base<IdxT>>>& override;

  auto list_sizes() noexcept
    -> raft::device_vector_view<uint32_t, uint32_t, raft::row_major> override;
  [[nodiscard]] auto list_sizes() const noexcept
    -> raft::device_vector_view<const uint32_t, uint32_t, raft::row_major> override;

  auto data_ptrs() noexcept
    -> raft::device_vector_view<uint8_t*, uint32_t, raft::row_major> override;
  [[nodiscard]] auto data_ptrs() const noexcept
    -> raft::device_vector_view<const uint8_t* const, uint32_t, raft::row_major> override;

  auto inds_ptrs() noexcept -> raft::device_vector_view<IdxT*, uint32_t, raft::row_major> override;
  [[nodiscard]] auto inds_ptrs() const noexcept
    -> raft::device_vector_view<const IdxT* const, uint32_t, raft::row_major> override;

  auto accum_sorted_sizes() noexcept
    -> raft::host_vector_view<IdxT, uint32_t, raft::row_major> override;
  [[nodiscard]] auto accum_sorted_sizes() const noexcept
    -> raft::host_vector_view<const IdxT, uint32_t, raft::row_major> override;

  [[nodiscard]] auto rotation_matrix_int8(const raft::resources& res) const
    -> raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> override;
  [[nodiscard]] auto rotation_matrix_half(const raft::resources& res) const
    -> raft::device_matrix_view<const half, uint32_t, raft::row_major> override;
  [[nodiscard]] auto centers_int8(const raft::resources& res) const
    -> raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> override;
  [[nodiscard]] auto centers_half(const raft::resources& res) const
    -> raft::device_matrix_view<const half, uint32_t, raft::row_major> override;

  [[nodiscard]] auto get_list_size_in_bytes(uint32_t label) const -> uint32_t override;

 protected:
  cuvs::distance::DistanceType metric_;
  codebook_gen codebook_kind_;
  list_layout codes_layout_;
  uint32_t dim_;
  uint32_t pq_bits_;
  uint32_t pq_dim_;
  bool conservative_memory_allocation_;

  // Primary data members
  std::vector<std::shared_ptr<list_data_base<IdxT>>> lists_;
  raft::device_vector<uint32_t, uint32_t, raft::row_major> list_sizes_;

  // Lazy-initialized low-precision variants of index members - for low-precision coarse search.
  // These are never serialized and not touched during build/extend.
  mutable std::optional<raft::device_matrix<int8_t, uint32_t, raft::row_major>> centers_int8_;
  mutable std::optional<raft::device_matrix<half, uint32_t, raft::row_major>> centers_half_;
  mutable std::optional<raft::device_matrix<int8_t, uint32_t, raft::row_major>>
    rotation_matrix_int8_;
  mutable std::optional<raft::device_matrix<half, uint32_t, raft::row_major>> rotation_matrix_half_;

  // Computed members for accelerating search.
  raft::device_vector<uint8_t*, uint32_t, raft::row_major> data_ptrs_;
  raft::device_vector<IdxT*, uint32_t, raft::row_major> inds_ptrs_;
  raft::host_vector<IdxT, uint32_t, raft::row_major> accum_sorted_sizes_;

  /** Throw an error if the index content is inconsistent. */
  void check_consistency();
};

template <typename IdxT>
class owning_impl : public index_impl<IdxT> {
 public:
  owning_impl(raft::resources const& handle,
              cuvs::distance::DistanceType metric,
              codebook_gen codebook_kind,
              uint32_t n_lists,
              uint32_t dim,
              uint32_t pq_bits,
              uint32_t pq_dim,
              bool conservative_memory_allocation,
              list_layout codes_layout = list_layout::INTERLEAVED);

  ~owning_impl()                                     = default;
  owning_impl(owning_impl&&)                         = default;
  auto operator=(owning_impl&&) -> owning_impl&      = default;
  owning_impl(const owning_impl&)                    = delete;
  auto operator=(const owning_impl&) -> owning_impl& = delete;

  auto pq_centers() noexcept -> raft::device_mdspan<float, pq_centers_extents, raft::row_major>;
  [[nodiscard]] auto pq_centers() const noexcept
    -> raft::device_mdspan<const float, pq_centers_extents, raft::row_major> override;

  auto centers() noexcept -> raft::device_matrix_view<float, uint32_t, raft::row_major>;
  [[nodiscard]] auto centers() const noexcept
    -> raft::device_matrix_view<const float, uint32_t, raft::row_major> override;

  auto centers_rot() noexcept -> raft::device_matrix_view<float, uint32_t, raft::row_major>;
  [[nodiscard]] auto centers_rot() const noexcept
    -> raft::device_matrix_view<const float, uint32_t, raft::row_major> override;

  auto rotation_matrix() noexcept -> raft::device_matrix_view<float, uint32_t, raft::row_major>;
  [[nodiscard]] auto rotation_matrix() const noexcept
    -> raft::device_matrix_view<const float, uint32_t, raft::row_major> override;

 private:
  raft::device_mdarray<float, pq_centers_extents, raft::row_major> pq_centers_;
  raft::device_matrix<float, uint32_t, raft::row_major> centers_;
  raft::device_matrix<float, uint32_t, raft::row_major> centers_rot_;
  raft::device_matrix<float, uint32_t, raft::row_major> rotation_matrix_;
};

template <typename IdxT>
class view_impl : public index_impl<IdxT> {
 public:
  view_impl(raft::resources const& handle,
            cuvs::distance::DistanceType metric,
            codebook_gen codebook_kind,
            uint32_t n_lists,
            uint32_t dim,
            uint32_t pq_bits,
            uint32_t pq_dim,
            bool conservative_memory_allocation,
            raft::device_mdspan<const float, pq_centers_extents, raft::row_major> pq_centers_view,
            raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_view,
            raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot_view,
            raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix_view,
            list_layout codes_layout = list_layout::INTERLEAVED);

  ~view_impl()                                   = default;
  view_impl(view_impl&&)                         = default;
  auto operator=(view_impl&&) -> view_impl&      = default;
  view_impl(const view_impl&)                    = delete;
  auto operator=(const view_impl&) -> view_impl& = delete;

  [[nodiscard]] auto pq_centers() const noexcept
    -> raft::device_mdspan<const float, pq_centers_extents, raft::row_major> override;

  [[nodiscard]] auto centers() const noexcept
    -> raft::device_matrix_view<const float, uint32_t, raft::row_major> override;

  [[nodiscard]] auto centers_rot() const noexcept
    -> raft::device_matrix_view<const float, uint32_t, raft::row_major> override;

  [[nodiscard]] auto rotation_matrix() const noexcept
    -> raft::device_matrix_view<const float, uint32_t, raft::row_major> override;

 private:
  raft::device_mdspan<const float, pq_centers_extents, raft::row_major> pq_centers_view_;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_view_;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot_view_;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix_view_;
};

}  // namespace cuvs::neighbors::ivf_pq
