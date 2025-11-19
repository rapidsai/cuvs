/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
              bool conservative_memory_allocation);

  ~index_impl() = default;

  cuvs::distance::DistanceType metric() const noexcept override;
  codebook_gen codebook_kind() const noexcept override;
  uint32_t dim() const noexcept override;
  uint32_t dim_ext() const noexcept override;
  uint32_t rot_dim() const noexcept override;
  uint32_t pq_bits() const noexcept override;
  uint32_t pq_dim() const noexcept override;
  uint32_t pq_len() const noexcept override;
  uint32_t pq_book_size() const noexcept override;
  uint32_t n_lists() const noexcept override;
  bool conservative_memory_allocation() const noexcept override;

  std::vector<std::shared_ptr<list_data<IdxT>>>& lists() noexcept override;
  const std::vector<std::shared_ptr<list_data<IdxT>>>& lists() const noexcept override;

  raft::device_vector_view<uint32_t, uint32_t, raft::row_major> list_sizes() noexcept override;
  raft::device_vector_view<const uint32_t, uint32_t, raft::row_major> list_sizes()
    const noexcept override;

  raft::device_vector_view<uint8_t*, uint32_t, raft::row_major> data_ptrs() noexcept override;
  raft::device_vector_view<const uint8_t* const, uint32_t, raft::row_major> data_ptrs()
    const noexcept override;

  raft::device_vector_view<IdxT*, uint32_t, raft::row_major> inds_ptrs() noexcept override;
  raft::device_vector_view<const IdxT* const, uint32_t, raft::row_major> inds_ptrs()
    const noexcept override;

  raft::host_vector_view<IdxT, uint32_t, raft::row_major> accum_sorted_sizes() noexcept override;
  raft::host_vector_view<const IdxT, uint32_t, raft::row_major> accum_sorted_sizes()
    const noexcept override;

  raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> rotation_matrix_int8(
    const raft::resources& res) const override;
  raft::device_matrix_view<const half, uint32_t, raft::row_major> rotation_matrix_half(
    const raft::resources& res) const override;
  raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> centers_int8(
    const raft::resources& res) const override;
  raft::device_matrix_view<const half, uint32_t, raft::row_major> centers_half(
    const raft::resources& res) const override;

  uint32_t get_list_size_in_bytes(uint32_t label) const noexcept override;

 protected:
  cuvs::distance::DistanceType metric_;
  codebook_gen codebook_kind_;
  uint32_t dim_;
  uint32_t pq_bits_;
  uint32_t pq_dim_;
  bool conservative_memory_allocation_;

  std::vector<std::shared_ptr<list_data<IdxT>>> lists_;
  raft::device_vector<uint32_t, uint32_t, raft::row_major> list_sizes_;
  raft::device_vector<uint8_t*, uint32_t, raft::row_major> data_ptrs_;
  raft::device_vector<IdxT*, uint32_t, raft::row_major> inds_ptrs_;
  raft::host_vector<IdxT, uint32_t, raft::row_major> accum_sorted_sizes_;

  mutable std::optional<raft::device_matrix<int8_t, uint32_t, raft::row_major>> centers_int8_;
  mutable std::optional<raft::device_matrix<half, uint32_t, raft::row_major>> centers_half_;
  mutable std::optional<raft::device_matrix<int8_t, uint32_t, raft::row_major>>
    rotation_matrix_int8_;
  mutable std::optional<raft::device_matrix<half, uint32_t, raft::row_major>> rotation_matrix_half_;
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
              bool conservative_memory_allocation);

  ~owning_impl();

  raft::device_mdspan<float, pq_centers_extents, raft::row_major>
  pq_centers() noexcept override;
  raft::device_mdspan<const float, pq_centers_extents, raft::row_major>
  pq_centers() const noexcept override;

  raft::device_matrix_view<float, uint32_t, raft::row_major> centers() noexcept override;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers()
    const noexcept override;

  raft::device_matrix_view<float, uint32_t, raft::row_major> centers_rot() noexcept override;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot()
    const noexcept override;

  raft::device_matrix_view<float, uint32_t, raft::row_major> rotation_matrix() noexcept override;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix()
    const noexcept override;

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
            raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix_view);

  ~view_impl() = default;

  raft::device_mdspan<float, pq_centers_extents, raft::row_major>
  pq_centers() noexcept override;
  raft::device_mdspan<const float, pq_centers_extents, raft::row_major>
  pq_centers() const noexcept override;

  raft::device_matrix_view<float, uint32_t, raft::row_major> centers() noexcept override;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers()
    const noexcept override;

  raft::device_matrix_view<float, uint32_t, raft::row_major> centers_rot() noexcept override;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot()
    const noexcept override;

  raft::device_matrix_view<float, uint32_t, raft::row_major> rotation_matrix() noexcept override;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix()
    const noexcept override;

 private:
  raft::device_mdspan<const float, pq_centers_extents, raft::row_major>
    pq_centers_view_;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_view_;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot_view_;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix_view_;
};

}  // namespace cuvs::neighbors::ivf_pq

