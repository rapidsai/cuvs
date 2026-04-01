/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.hpp"
#include <cstdint>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>

namespace cuvs::neighbors::ivf_sq {

/**
 * @defgroup ivf_sq_cpp_index_params IVF-SQ index build parameters
 * @{
 */

constexpr static uint32_t kIndexGroupSize = 32;

struct index_params : cuvs::neighbors::index_params {
  uint32_t n_lists                    = 1024;
  uint32_t kmeans_n_iters             = 20;
  double kmeans_trainset_fraction     = 0.5;
  bool adaptive_centers               = false;
  bool conservative_memory_allocation = false;
  bool add_data_on_build              = true;
};

struct search_params : cuvs::neighbors::search_params {
  uint32_t n_probes = 20;
};

static_assert(std::is_aggregate_v<index_params>);
static_assert(std::is_aggregate_v<search_params>);

/**
 * @}
 */

/**
 * @defgroup ivf_sq_cpp_list_spec IVF-SQ list storage spec
 * @{
 */

template <typename SizeT, typename IdxT, typename ExtT>
struct list_spec {
  static_assert(std::is_same_v<IdxT, uint8_t>, "IVF-SQ code type IdxT must be uint8_t");

  using value_type   = IdxT;
  using list_extents = raft::matrix_extent<SizeT>;
  using index_type   = ExtT;

  SizeT align_max;
  SizeT align_min;
  uint32_t dim;

  constexpr list_spec(uint32_t dim, bool conservative_memory_allocation)
    : dim(dim),
      align_min(kIndexGroupSize),
      align_max(conservative_memory_allocation ? kIndexGroupSize : 1024)
  {
  }

  template <typename OtherSizeT>
  constexpr explicit list_spec(const list_spec<OtherSizeT, IdxT, ExtT>& other_spec)
    : dim{other_spec.dim}, align_min{other_spec.align_min}, align_max{other_spec.align_max}
  {
  }

  static constexpr uint32_t kVecLen = 16;

  constexpr auto make_list_extents(SizeT n_rows) const -> list_extents
  {
    uint32_t padded = ((dim + kVecLen - 1) / kVecLen) * kVecLen;
    return raft::make_extents<SizeT>(n_rows, padded);
  }
};

template <typename IdxT, typename ExtT, typename SizeT = uint32_t>
using list_data = ivf::list<list_spec, SizeT, IdxT, ExtT>;

/**
 * @}
 */

/**
 * @defgroup ivf_sq_cpp_index IVF-SQ index
 * @{
 */

/**
 * @brief IVF-SQ index.
 *
 * @tparam IdxT  SQ code type. Only uint8_t (8-bit, codes in [0,255]) for now.
 *
 * No member depends on the raw data type T (float, half). T appears only
 * in the free-function signatures (build, search, extend) where input data
 * is consumed, following the IVF-PQ pattern.
 */
template <typename IdxT>
struct index : cuvs::neighbors::index {
  static_assert(std::is_same_v<IdxT, uint8_t>, "IVF-SQ code type IdxT must be uint8_t for now.");

  using index_params_type  = ivf_sq::index_params;
  using search_params_type = ivf_sq::search_params;
  using code_type          = IdxT;

  static constexpr uint32_t sq_bits = sizeof(IdxT) * 8;

 public:
  index(const index&)            = delete;
  index(index&&)                 = default;
  index& operator=(const index&) = delete;
  index& operator=(index&&)      = default;
  ~index()                       = default;

  index(raft::resources const& res);
  index(raft::resources const& res, const index_params& params, uint32_t dim);
  index(raft::resources const& res,
        cuvs::distance::DistanceType metric,
        uint32_t n_lists,
        uint32_t dim,
        bool adaptive_centers,
        bool conservative_memory_allocation);

  cuvs::distance::DistanceType metric() const noexcept;
  bool adaptive_centers() const noexcept;
  int64_t size() const noexcept;
  uint32_t dim() const noexcept;
  uint32_t n_lists() const noexcept;
  bool conservative_memory_allocation() const noexcept;

  raft::device_vector_view<uint32_t, uint32_t> list_sizes() noexcept;
  raft::device_vector_view<const uint32_t, uint32_t> list_sizes() const noexcept;

  raft::device_matrix_view<float, uint32_t, raft::row_major> centers() noexcept;
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers() const noexcept;

  std::optional<raft::device_vector_view<float, uint32_t>> center_norms() noexcept;
  std::optional<raft::device_vector_view<const float, uint32_t>> center_norms() const noexcept;
  void allocate_center_norms(raft::resources const& res);

  raft::device_vector_view<float, uint32_t> sq_vmin() noexcept;
  raft::device_vector_view<const float, uint32_t> sq_vmin() const noexcept;

  raft::device_vector_view<float, uint32_t> sq_delta() noexcept;
  raft::device_vector_view<const float, uint32_t> sq_delta() const noexcept;

  raft::host_vector_view<int64_t, uint32_t> accum_sorted_sizes() noexcept;
  [[nodiscard]] raft::host_vector_view<const int64_t, uint32_t> accum_sorted_sizes() const noexcept;

  raft::device_vector_view<IdxT*, uint32_t> data_ptrs() noexcept;
  raft::device_vector_view<IdxT* const, uint32_t> data_ptrs() const noexcept;

  raft::device_vector_view<int64_t*, uint32_t> inds_ptrs() noexcept;
  raft::device_vector_view<int64_t* const, uint32_t> inds_ptrs() const noexcept;

  std::vector<std::shared_ptr<list_data<IdxT, int64_t>>>& lists() noexcept;
  const std::vector<std::shared_ptr<list_data<IdxT, int64_t>>>& lists() const noexcept;

  void check_consistency();

 private:
  cuvs::distance::DistanceType metric_;
  bool adaptive_centers_;
  bool conservative_memory_allocation_;

  std::vector<std::shared_ptr<list_data<IdxT, int64_t>>> lists_;
  raft::device_vector<uint32_t, uint32_t> list_sizes_;
  raft::device_matrix<float, uint32_t, raft::row_major> centers_;
  std::optional<raft::device_vector<float, uint32_t>> center_norms_;
  raft::device_vector<float, uint32_t> sq_vmin_;
  raft::device_vector<float, uint32_t> sq_delta_;

  raft::device_vector<IdxT*, uint32_t> data_ptrs_;
  raft::device_vector<int64_t*, uint32_t> inds_ptrs_;
  raft::host_vector<int64_t, uint32_t> accum_sorted_sizes_;
};

/**
 * @}
 */

/**
 * @defgroup ivf_sq_cpp_index_build IVF-SQ index build
 * @{
 */

auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_sq::index_params& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_sq::index<uint8_t>;

void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_sq::index_params& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_sq::index<uint8_t>& idx);

auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_sq::index_params& index_params,
           raft::device_matrix_view<const half, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_sq::index<uint8_t>;

void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_sq::index_params& index_params,
           raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_sq::index<uint8_t>& idx);

auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_sq::index_params& index_params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_sq::index<uint8_t>;

void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_sq::index_params& index_params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_sq::index<uint8_t>& idx);

auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_sq::index_params& index_params,
           raft::host_matrix_view<const half, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_sq::index<uint8_t>;

void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_sq::index_params& index_params,
           raft::host_matrix_view<const half, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_sq::index<uint8_t>& idx);

/**
 * @}
 */

/**
 * @defgroup ivf_sq_cpp_index_extend IVF-SQ index extend
 * @{
 */

auto extend(raft::resources const& handle,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_sq::index<uint8_t>& orig_index)
  -> cuvs::neighbors::ivf_sq::index<uint8_t>;

void extend(raft::resources const& handle,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_sq::index<uint8_t>* idx);

auto extend(raft::resources const& handle,
            raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_sq::index<uint8_t>& orig_index)
  -> cuvs::neighbors::ivf_sq::index<uint8_t>;

void extend(raft::resources const& handle,
            raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_sq::index<uint8_t>* idx);

auto extend(raft::resources const& handle,
            raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_sq::index<uint8_t>& orig_index)
  -> cuvs::neighbors::ivf_sq::index<uint8_t>;

void extend(raft::resources const& handle,
            raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_sq::index<uint8_t>* idx);

auto extend(raft::resources const& handle,
            raft::host_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_sq::index<uint8_t>& orig_index)
  -> cuvs::neighbors::ivf_sq::index<uint8_t>;

void extend(raft::resources const& handle,
            raft::host_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_sq::index<uint8_t>* idx);

/**
 * @}
 */

/**
 * @defgroup ivf_sq_cpp_index_search IVF-SQ index search
 * @{
 */

void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_sq::search_params& params,
            const cuvs::neighbors::ivf_sq::index<uint8_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_sq::search_params& params,
            const cuvs::neighbors::ivf_sq::index<uint8_t>& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @}
 */

/**
 * @defgroup ivf_sq_cpp_index_serialize IVF-SQ index serialize
 * @{
 */

void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::ivf_sq::index<uint8_t>& index);

void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_sq::index<uint8_t>* index);

/**
 * @}
 */

}  // namespace cuvs::neighbors::ivf_sq
