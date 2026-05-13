/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>   // get_device_for_address
#include <raft/util/integer_utils.hpp>  // rounding up

#include <cuvs/core/bitmap.hpp>
#include <cuvs/core/bitset.hpp>
#include <raft/core/detail/macros.hpp>

#include <cuda_fp16.h>

#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <variant>

#ifdef __cpp_lib_bitops
#include <bit>
#include <cuvs/core/export.hpp>
#endif

namespace CUVS_EXPORT cuvs {
namespace neighbors {
/**
 * @addtogroup cagra_cpp_index_params
 * @{
 */

/* Graph build algo used in cagra and all_neighbors */
enum GRAPH_BUILD_ALGO { BRUTE_FORCE = 0, IVF_PQ = 1, NN_DESCENT = 2, ACE = 3 };

/** Parameters for VPQ compression. */
struct vpq_params {
  /**
   * The bit length of the vector element after compression by PQ.
   *
   * Possible values: [4, 5, 6, 7, 8].
   *
   * Hint: the smaller the 'pq_bits', the smaller the index size and the better the search
   * performance, but the lower the recall.
   */
  uint32_t pq_bits = 8;
  /**
   * The dimensionality of the vector after compression by PQ.
   * When zero, an optimal value is selected using a heuristic.
   *
   * TODO: at the moment `dim` must be a multiple `pq_dim`.
   */
  uint32_t pq_dim = 0;
  /**
   * Vector Quantization (VQ) codebook size - number of "coarse cluster centers".
   * When zero, an optimal value is selected using a heuristic.
   */
  uint32_t vq_n_centers = 0;
  /** The number of iterations searching for kmeans centers (both VQ & PQ phases). */
  uint32_t kmeans_n_iters = 25;
  /**
   * The fraction of data to use during iterative kmeans building (VQ phase).
   * When zero, an optimal value is selected using a heuristic.
   * @deprecated Prefer using `max_train_points_per_vq_cluster` instead.
   */
  double vq_kmeans_trainset_fraction = 0;
  /**
   * The fraction of data to use during iterative kmeans building (PQ phase).
   * When zero, an optimal value is selected using a heuristic.
   * @deprecated Prefer using `max_train_points_per_pq_code` instead.
   */
  double pq_kmeans_trainset_fraction = 0;
  /**
   * Type of k-means algorithm for PQ training.
   * Balanced k-means tends to be faster than regular k-means for PQ training, for
   * problem sets where the number of points per cluster are approximately equal.
   * Regular k-means may be better for skewed cluster distributions.
   */
  cuvs::cluster::kmeans::kmeans_type pq_kmeans_type =
    cuvs::cluster::kmeans::kmeans_type::KMeansBalanced;
  /**
   * The max number of data points to use per PQ code during PQ codebook training. Using more data
   * points per PQ code may increase the quality of PQ codebook but may also increase the build
   * time. We will use `pq_n_centers * max_train_points_per_pq_code` training
   * points to train each PQ codebook.
   */
  uint32_t max_train_points_per_pq_code = 256;
  /**
   * The max number of data points to use per VQ cluster during training.
   */
  uint32_t max_train_points_per_vq_cluster = 1024;
};

/** @} */  // end group cagra_cpp_index_params

/**
 * @defgroup neighbors_index Approximate Nearest Neighbors Types
 * @{
 */

/** The base for approximate KNN index structures. */
struct index {};

/** The base for KNN index parameters. */
struct index_params {
  /** Distance type. */
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
  /** The argument used by some distance metrics. */
  float metric_arg = 2.0f;
};

struct search_params {};

/**
 * @brief Strategy for merging indices.
 *
 * This enum is declared separately to avoid namespace pollution when including common.hpp.
 * It provides a generic merge strategy that can be used across different index types.
 */
enum class MergeStrategy {
  /** Merge indices physically by combining their data structures */
  MERGE_STRATEGY_PHYSICAL = 0,
  /** Merge indices logically by creating a composite wrapper */
  MERGE_STRATEGY_LOGICAL = 1
};

/** @} */  // end group neighbors_index

/**
 * @brief Tags selecting dataset representation for `dataset` / `dataset_view`.
 *
 * The first template parameter `containertype` on `dataset` / `dataset_view` is one of these types.
 */
struct empty_dataset_container {};
struct padded_dataset_container {};
struct vpq_dataset_container {};
struct strided_dataset_container {};
/**
 * Tag for owning dataset unions (`any_owning_dataset<IdxT>`).
 *
 * The specialization `dataset<any_owning_dataset_container, void, IdxT>` lists several
 * `dataset<..., DataT, IdxT>` alternatives with different `DataT` (float/half/int8/uint8 padded,
 * VPQ codebook element types). There is no single outer `DataT` template parameter for the wrapper:
 * which variant alternative is active is often chosen when loading from disk or wiring ownership,
 * while many call sites keep one nominal type `any_owning_dataset<IdxT>` without fixing element
 * type at compile time.
 */
struct any_owning_dataset_container {};
/** Tag: non-owning view union (`any_dataset_view<DataT, IdxT>`). */
struct any_dataset_view_container {};

template <typename containertype,
          typename DataT,
          typename IdxT,
          bool is_device_accessible,
          bool is_host_accessible>
struct dataset {
  static_assert(!std::is_same_v<containertype, containertype>,
                "dataset: unsupported containertype / type-parameter combination");
};

template <typename containertype,
          typename DataT,
          typename IdxT,
          bool is_device_accessible,
          bool is_host_accessible>
struct dataset_view {
  static_assert(!std::is_same_v<containertype, containertype>,
                "dataset_view: unsupported containertype / type-parameter combination");
};

// -----------------------------------------------------------------------------
// empty
// -----------------------------------------------------------------------------

template <typename IdxT>
struct dataset<empty_dataset_container, void, IdxT, false, false> {
  using index_type = IdxT;
  uint32_t suggested_dim{};
  explicit dataset(uint32_t dim) noexcept : suggested_dim(dim) {}
  [[nodiscard]] auto n_rows() const noexcept -> index_type { return 0; }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return suggested_dim; }
};

template <typename IdxT>
struct dataset_view<empty_dataset_container, void, IdxT, false, false> {
  using index_type = IdxT;
  uint32_t suggested_dim_{};
  explicit dataset_view(uint32_t dim) noexcept : suggested_dim_(dim) {}
  [[nodiscard]] auto n_rows() const noexcept -> index_type { return 0; }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return suggested_dim_; }
};

// -----------------------------------------------------------------------------
// padded (device row-major with logical dim vs stride)
// -----------------------------------------------------------------------------

template <typename DataT, typename IdxT>
struct dataset<padded_dataset_container, DataT, IdxT, true, false> {
  using index_type   = IdxT;
  using value_type   = DataT;
  using storage_type = raft::device_matrix<value_type, index_type, raft::row_major>;
  using view_type    = raft::device_matrix_view<const value_type, index_type, raft::row_major>;

  storage_type data_;
  uint32_t dim_;

  dataset(storage_type&& data, uint32_t logical_dim) noexcept
    : data_{std::move(data)}, dim_{logical_dim}
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> index_type { return data_.extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return dim_; }
  [[nodiscard]] auto stride() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(data_.extent(1));
  }
  [[nodiscard]] auto view() const noexcept -> view_type { return data_.view(); }
  [[nodiscard]] auto as_dataset_view() const noexcept
    -> dataset_view<padded_dataset_container, DataT, IdxT, true, false>
  {
    return dataset_view<padded_dataset_container, DataT, IdxT, true, false>(data_.view(), dim_);
  }
  [[nodiscard]] auto data_handle() noexcept -> value_type* { return data_.data_handle(); }
  [[nodiscard]] auto data_handle() const noexcept -> const value_type*
  {
    return data_.data_handle();
  }
};

template <typename DataT, typename IdxT>
struct dataset_view<padded_dataset_container, DataT, IdxT, true, false> {
  using index_type = IdxT;
  using value_type = DataT;
  using view_type  = raft::device_matrix_view<const value_type, index_type, raft::row_major>;

  view_type data_;
  uint32_t logical_dim_;

  explicit dataset_view(view_type v) noexcept
    : data_(v), logical_dim_(static_cast<uint32_t>(v.extent(1)))
  {
  }

  dataset_view(view_type v, uint32_t logical_dim) noexcept : data_(v), logical_dim_(logical_dim) {}

  dataset_view(dataset_view const& other) noexcept
    : data_(other.data_), logical_dim_(other.logical_dim_)
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> index_type { return data_.extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return logical_dim_; }
  [[nodiscard]] auto stride() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(data_.stride(0) > 0 ? data_.stride(0) : data_.extent(1));
  }
  [[nodiscard]] auto view() const noexcept -> view_type { return data_; }
};

// -----------------------------------------------------------------------------
// VPQ compressed owning dataset (+ non-owning view below)
// -----------------------------------------------------------------------------
//
// Owning block is first for file organization. `dataset_view<vpq_dataset_container, …>` is
// forward-declared so `as_dataset_view()` can be declared here; its definition (and the view’s
// constructor body that wraps `this`) come after the full view specialization.

template <typename DataT, typename IdxT>
struct dataset_view<vpq_dataset_container, DataT, IdxT, true, false>;

template <typename DataT, typename IdxT>
struct dataset<vpq_dataset_container, DataT, IdxT, true, false> {
  using index_type = IdxT;
  /** Same as `DataT`: floating-point type used for VQ/PQ codebooks (rows are still uint8 codes). */
  using math_type = DataT;
  raft::device_matrix<math_type, uint32_t, raft::row_major> vq_code_book;
  raft::device_matrix<math_type, uint32_t, raft::row_major> pq_code_book;
  raft::device_matrix<uint8_t, index_type, raft::row_major> data;

  dataset(raft::device_matrix<math_type, uint32_t, raft::row_major>&& vq_code_book,
          raft::device_matrix<math_type, uint32_t, raft::row_major>&& pq_code_book,
          raft::device_matrix<uint8_t, index_type, raft::row_major>&& data)
    : vq_code_book{std::move(vq_code_book)},
      pq_code_book{std::move(pq_code_book)},
      data{std::move(data)}
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> index_type { return data.extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return vq_code_book.extent(1); }

  [[nodiscard]] constexpr inline auto encoded_row_length() const noexcept -> uint32_t
  {
    return data.extent(1);
  }
  [[nodiscard]] constexpr inline auto vq_n_centers() const noexcept -> uint32_t
  {
    return vq_code_book.extent(0);
  }
  [[nodiscard]] constexpr inline auto pq_bits() const noexcept -> uint32_t
  {
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
  [[nodiscard]] constexpr inline auto pq_dim() const noexcept -> uint32_t
  {
    return raft::div_rounding_up_unsafe(dim(), pq_len());
  }
  [[nodiscard]] constexpr inline auto pq_len() const noexcept -> uint32_t
  {
    return pq_code_book.extent(1);
  }
  [[nodiscard]] constexpr inline auto pq_n_centers() const noexcept -> uint32_t
  {
    return pq_code_book.extent(0);
  }

  /** Non-owning view for storing in `any_dataset_view` (same role as
   * `padded_dataset::as_dataset_view`). */
  [[nodiscard]] auto as_dataset_view() const noexcept
    -> dataset_view<vpq_dataset_container, DataT, IdxT, true, false>;
};

// -----------------------------------------------------------------------------
// VPQ non-owning device view (pointer to `vpq_dataset`; same `vpq_dataset_container` tag as
// owning).
// -----------------------------------------------------------------------------

template <typename DataT, typename IdxT>
struct dataset_view<vpq_dataset_container, DataT, IdxT, true, false> {
  using index_type  = IdxT;
  using target_type = dataset<vpq_dataset_container, DataT, IdxT, true, false>;

  target_type const* target_{};

  dataset_view() = default;

  explicit dataset_view(target_type const* ptr) noexcept : target_(ptr)
  {
    RAFT_EXPECTS(ptr != nullptr, "vpq_dataset_view: null target");
  }

  [[nodiscard]] auto n_rows() const noexcept -> index_type { return target_->n_rows(); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return target_->dim(); }
  [[nodiscard]] target_type const& dset() const noexcept { return *target_; }
};

template <typename DataT, typename IdxT>
[[nodiscard]] inline auto
dataset<vpq_dataset_container, DataT, IdxT, true, false>::as_dataset_view() const noexcept
  -> dataset_view<vpq_dataset_container, DataT, IdxT, true, false>
{
  return dataset_view<vpq_dataset_container, DataT, IdxT, true, false>(this);
}

// -----------------------------------------------------------------------------
// Strided owning device storage (`layout_stride` mdarray)
// -----------------------------------------------------------------------------

template <typename DataT, typename IdxT>
struct dataset<strided_dataset_container, DataT, IdxT, true, false> {
  using index_type   = IdxT;
  using value_type   = DataT;
  using view_type    = raft::device_matrix_view<const value_type, index_type, raft::layout_stride>;
  using storage_type = raft::device_matrix<value_type, index_type, raft::layout_stride>;
  using mapping_type = typename view_type::mapping_type;

  storage_type data;
  mapping_type view_mapping;

  dataset(storage_type&& store, mapping_type view_mapping) noexcept
    : data{std::move(store)}, view_mapping{std::move(view_mapping)}
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> index_type { return view().extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(view().extent(1));
  }
  [[nodiscard]] constexpr auto stride() const noexcept -> uint32_t
  {
    auto v = view();
    return static_cast<uint32_t>(v.stride(0) > 0 ? v.stride(0) : v.extent(1));
  }
  [[nodiscard]] auto view() const noexcept -> view_type
  {
    return view_type{data.data_handle(), view_mapping};
  }
};

// -----------------------------------------------------------------------------
// Strided non-owning device view
// -----------------------------------------------------------------------------

template <typename DataT, typename IdxT>
struct dataset_view<strided_dataset_container, DataT, IdxT, true, false> {
  using index_type = IdxT;
  using value_type = DataT;
  using view_type  = raft::device_matrix_view<const value_type, index_type, raft::layout_stride>;

  view_type data_;

  explicit dataset_view(view_type v) noexcept : data_(v) {}

  [[nodiscard]] auto n_rows() const noexcept -> index_type { return data_.extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(data_.extent(1));
  }
  [[nodiscard]] constexpr auto stride() const noexcept -> uint32_t
  {
    auto v = data_;
    return static_cast<uint32_t>(v.stride(0) > 0 ? v.stride(0) : v.extent(1));
  }
  [[nodiscard]] auto view() const noexcept -> view_type { return data_; }
};

/**
 * @brief Aliases for concrete `dataset` / `dataset_view` layouts.
 *
 * Kept in one place (after the last non-erased layout specialization) so the mapping from public
 * names to `dataset<containertype, …>` is easy to scan. These cannot be moved above the
 * specializations: the primary `dataset` / `dataset_view` templates are not defined for unknown
 * tags, and some bodies must spell `dataset_view<padded_dataset_container, …>` before
 * `padded_dataset_view` exists (see `dataset<padded_dataset_container>::as_dataset_view`).
 * VPQ: `dataset_view<vpq_dataset_container, …>` is forward-declared, then owning `dataset`, then
 * the full view specialization and `as_dataset_view()` out-of-line (constructor needs a complete
 * view type).
 *
 * Variant member helpers (`any_dataset_view_types`, `any_owning_dataset_types`) follow; see
 * section comments there.
 */
template <typename IdxT>
using empty_dataset = dataset<empty_dataset_container, void, IdxT, false, false>;

template <typename IdxT>
using empty_dataset_view = dataset_view<empty_dataset_container, void, IdxT, false, false>;

template <typename DataT, typename IdxT>
using padded_dataset = dataset<padded_dataset_container, DataT, IdxT, true, false>;

template <typename DataT, typename IdxT>
using padded_dataset_view = dataset_view<padded_dataset_container, DataT, IdxT, true, false>;

template <typename DataT, typename IdxT>
using device_padded_dataset = padded_dataset<DataT, IdxT>;

template <typename DataT, typename IdxT>
using device_padded_dataset_view = padded_dataset_view<DataT, IdxT>;

template <typename DataT, typename IdxT>
using vpq_dataset = dataset<vpq_dataset_container, DataT, IdxT, true, false>;

/** Non-owning view of a device `vpq_dataset` (codebooks + encoded rows). */
template <typename DataT, typename IdxT>
using vpq_dataset_view = dataset_view<vpq_dataset_container, DataT, IdxT, true, false>;

template <typename DataT, typename IdxT>
using strided_owning_dataset = dataset<strided_dataset_container, DataT, IdxT, true, false>;

/** Non-owning strided device rows (`layout_stride`). */
template <typename DataT, typename IdxT>
using strided_dataset_view = dataset_view<strided_dataset_container, DataT, IdxT, true, false>;

/**
 * Concrete types held by `any_dataset_view<DataT, IdxT>`'s `std::variant`. Dispatch with
 * `std::holds_alternative<T>` / `std::get<T>` on `view.as_variant()` using these aliases — no
 * parallel numeric tags.
 */
template <typename DataT, typename IdxT>
struct any_dataset_view_types {
  using empty_view   = empty_dataset_view<IdxT>;
  using vpq_f16_view = vpq_dataset_view<half, IdxT>;
  using vpq_f32_view = vpq_dataset_view<float, IdxT>;
  using padded_view  = padded_dataset_view<DataT, IdxT>;
  using strided_view = strided_dataset_view<DataT, IdxT>;
};

/**
 * Concrete types held by `any_owning_dataset<IdxT>`'s `std::variant`. Dispatch with
 * `std::holds_alternative<T>` / `std::get<T>` on `dataset.as_variant()`.
 *
 * Strided owning alternatives mirror element widths used for padded/VPQ paths. Blobs tagged
 * `kSerializeStridedDataset` deserialize into `strided_owning_dataset` (same row pitch `stride`
 * as on save when present in the stream). `serialize(any_owning_dataset)` uses the same payload as
 * non-owning `strided_dataset_view` for those variants.
 */
template <typename IdxT>
struct any_owning_dataset_types {
  using empty_owning       = empty_dataset<IdxT>;
  using padded_f32_owning  = padded_dataset<float, IdxT>;
  using padded_f16_owning  = padded_dataset<half, IdxT>;
  using padded_i8_owning   = padded_dataset<int8_t, IdxT>;
  using padded_u8_owning   = padded_dataset<uint8_t, IdxT>;
  using strided_f32_owning = strided_owning_dataset<float, IdxT>;
  using strided_f16_owning = strided_owning_dataset<half, IdxT>;
  using strided_i8_owning  = strided_owning_dataset<int8_t, IdxT>;
  using strided_u8_owning  = strided_owning_dataset<uint8_t, IdxT>;
  using vpq_f32_owning     = vpq_dataset<float, IdxT>;
  using vpq_f16_owning     = vpq_dataset<half, IdxT>;
};

// `void` second parameter: no universal row element type for the whole wrapper; each
// `owning_variant` member carries its own `DataT`. See comment on `any_owning_dataset_container`.
template <typename IdxT>
struct dataset<any_owning_dataset_container, void, IdxT, false, false> {
  using index_type     = IdxT;
  using owning_variant = std::variant<typename any_owning_dataset_types<IdxT>::empty_owning,
                                      typename any_owning_dataset_types<IdxT>::padded_f32_owning,
                                      typename any_owning_dataset_types<IdxT>::padded_f16_owning,
                                      typename any_owning_dataset_types<IdxT>::padded_i8_owning,
                                      typename any_owning_dataset_types<IdxT>::padded_u8_owning,
                                      typename any_owning_dataset_types<IdxT>::strided_f32_owning,
                                      typename any_owning_dataset_types<IdxT>::strided_f16_owning,
                                      typename any_owning_dataset_types<IdxT>::strided_i8_owning,
                                      typename any_owning_dataset_types<IdxT>::strided_u8_owning,
                                      typename any_owning_dataset_types<IdxT>::vpq_f32_owning,
                                      typename any_owning_dataset_types<IdxT>::vpq_f16_owning>;

  owning_variant storage_;

  dataset() = default;

  template <typename containertype,
            typename DataT,
            bool is_device_accessible,
            bool is_host_accessible>
  explicit dataset(
    dataset<containertype, DataT, IdxT, is_device_accessible, is_host_accessible>&& x)
    : storage_(std::move(x))
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> index_type
  {
    using OT = any_owning_dataset_types<IdxT>;
    if (std::holds_alternative<typename OT::empty_owning>(storage_)) {
      return std::get<typename OT::empty_owning>(storage_).n_rows();
    }
    if (std::holds_alternative<typename OT::padded_f32_owning>(storage_)) {
      return std::get<typename OT::padded_f32_owning>(storage_).n_rows();
    }
    if (std::holds_alternative<typename OT::padded_f16_owning>(storage_)) {
      return std::get<typename OT::padded_f16_owning>(storage_).n_rows();
    }
    if (std::holds_alternative<typename OT::padded_i8_owning>(storage_)) {
      return std::get<typename OT::padded_i8_owning>(storage_).n_rows();
    }
    if (std::holds_alternative<typename OT::padded_u8_owning>(storage_)) {
      return std::get<typename OT::padded_u8_owning>(storage_).n_rows();
    }
    if (std::holds_alternative<typename OT::strided_f32_owning>(storage_)) {
      return std::get<typename OT::strided_f32_owning>(storage_).n_rows();
    }
    if (std::holds_alternative<typename OT::strided_f16_owning>(storage_)) {
      return std::get<typename OT::strided_f16_owning>(storage_).n_rows();
    }
    if (std::holds_alternative<typename OT::strided_i8_owning>(storage_)) {
      return std::get<typename OT::strided_i8_owning>(storage_).n_rows();
    }
    if (std::holds_alternative<typename OT::strided_u8_owning>(storage_)) {
      return std::get<typename OT::strided_u8_owning>(storage_).n_rows();
    }
    if (std::holds_alternative<typename OT::vpq_f32_owning>(storage_)) {
      return std::get<typename OT::vpq_f32_owning>(storage_).n_rows();
    }
    if (std::holds_alternative<typename OT::vpq_f16_owning>(storage_)) {
      return std::get<typename OT::vpq_f16_owning>(storage_).n_rows();
    }
    return IdxT{};
  }

  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    using OT = any_owning_dataset_types<IdxT>;
    if (std::holds_alternative<typename OT::empty_owning>(storage_)) {
      return std::get<typename OT::empty_owning>(storage_).dim();
    }
    if (std::holds_alternative<typename OT::padded_f32_owning>(storage_)) {
      return std::get<typename OT::padded_f32_owning>(storage_).dim();
    }
    if (std::holds_alternative<typename OT::padded_f16_owning>(storage_)) {
      return std::get<typename OT::padded_f16_owning>(storage_).dim();
    }
    if (std::holds_alternative<typename OT::padded_i8_owning>(storage_)) {
      return std::get<typename OT::padded_i8_owning>(storage_).dim();
    }
    if (std::holds_alternative<typename OT::padded_u8_owning>(storage_)) {
      return std::get<typename OT::padded_u8_owning>(storage_).dim();
    }
    if (std::holds_alternative<typename OT::strided_f32_owning>(storage_)) {
      return std::get<typename OT::strided_f32_owning>(storage_).dim();
    }
    if (std::holds_alternative<typename OT::strided_f16_owning>(storage_)) {
      return std::get<typename OT::strided_f16_owning>(storage_).dim();
    }
    if (std::holds_alternative<typename OT::strided_i8_owning>(storage_)) {
      return std::get<typename OT::strided_i8_owning>(storage_).dim();
    }
    if (std::holds_alternative<typename OT::strided_u8_owning>(storage_)) {
      return std::get<typename OT::strided_u8_owning>(storage_).dim();
    }
    if (std::holds_alternative<typename OT::vpq_f32_owning>(storage_)) {
      return std::get<typename OT::vpq_f32_owning>(storage_).dim();
    }
    if (std::holds_alternative<typename OT::vpq_f16_owning>(storage_)) {
      return std::get<typename OT::vpq_f16_owning>(storage_).dim();
    }
    return 0;
  }

  [[nodiscard]] owning_variant const& as_variant() const noexcept { return storage_; }
  [[nodiscard]] owning_variant& as_variant() noexcept { return storage_; }
};

template <typename DataT, typename IdxT>
struct dataset_view<any_dataset_view_container, DataT, IdxT, true, false> {
  using index_type   = IdxT;
  using variant_type = std::variant<typename any_dataset_view_types<DataT, IdxT>::empty_view,
                                    typename any_dataset_view_types<DataT, IdxT>::vpq_f16_view,
                                    typename any_dataset_view_types<DataT, IdxT>::vpq_f32_view,
                                    typename any_dataset_view_types<DataT, IdxT>::padded_view,
                                    typename any_dataset_view_types<DataT, IdxT>::strided_view>;

  variant_type storage_;

  dataset_view() = default;

  /** Non-explicit conversions so `device_padded_dataset_view` / VPQ / strided / empty views bind to
   *  APIs taking `any_dataset_view` without manual wrapping. */
  dataset_view(typename any_dataset_view_types<DataT, IdxT>::empty_view const& v) : storage_(v) {}
  dataset_view(typename any_dataset_view_types<DataT, IdxT>::vpq_f16_view const& v) : storage_(v) {}
  dataset_view(typename any_dataset_view_types<DataT, IdxT>::vpq_f32_view const& v) : storage_(v) {}
  dataset_view(typename any_dataset_view_types<DataT, IdxT>::padded_view const& v) : storage_(v) {}
  dataset_view(typename any_dataset_view_types<DataT, IdxT>::strided_view const& v) : storage_(v) {}

  template <typename Alt>
  explicit dataset_view(Alt&& alt) : storage_(std::forward<Alt>(alt))
  {
  }

  explicit dataset_view(variant_type v) : storage_(std::move(v)) {}

  [[nodiscard]] auto n_rows() const noexcept -> index_type
  {
    using VT = any_dataset_view_types<DataT, IdxT>;
    if (std::holds_alternative<typename VT::empty_view>(storage_)) {
      return std::get<typename VT::empty_view>(storage_).n_rows();
    }
    if (std::holds_alternative<typename VT::vpq_f16_view>(storage_)) {
      return std::get<typename VT::vpq_f16_view>(storage_).n_rows();
    }
    if (std::holds_alternative<typename VT::vpq_f32_view>(storage_)) {
      return std::get<typename VT::vpq_f32_view>(storage_).n_rows();
    }
    if (std::holds_alternative<typename VT::padded_view>(storage_)) {
      return std::get<typename VT::padded_view>(storage_).n_rows();
    }
    if (std::holds_alternative<typename VT::strided_view>(storage_)) {
      return std::get<typename VT::strided_view>(storage_).n_rows();
    }
    return IdxT{};
  }

  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    using VT = any_dataset_view_types<DataT, IdxT>;
    if (std::holds_alternative<typename VT::empty_view>(storage_)) {
      return std::get<typename VT::empty_view>(storage_).dim();
    }
    if (std::holds_alternative<typename VT::vpq_f16_view>(storage_)) {
      return std::get<typename VT::vpq_f16_view>(storage_).dim();
    }
    if (std::holds_alternative<typename VT::vpq_f32_view>(storage_)) {
      return std::get<typename VT::vpq_f32_view>(storage_).dim();
    }
    if (std::holds_alternative<typename VT::padded_view>(storage_)) {
      return std::get<typename VT::padded_view>(storage_).dim();
    }
    if (std::holds_alternative<typename VT::strided_view>(storage_)) {
      return std::get<typename VT::strided_view>(storage_).dim();
    }
    return 0;
  }

  [[nodiscard]] variant_type const& as_variant() const noexcept { return storage_; }
  [[nodiscard]] variant_type& as_variant() noexcept { return storage_; }
};

// -----------------------------------------------------------------------------
// Type-erased / union aliases — non-owning view union and owning variant typedefs
// -----------------------------------------------------------------------------

template <typename DataT, typename IdxT>
using any_dataset_view = dataset_view<any_dataset_view_container, DataT, IdxT, true, false>;

/** Owning union for deserialize / transport; see `any_owning_dataset_container`. */
template <typename IdxT>
using any_owning_dataset = dataset<any_owning_dataset_container, void, IdxT, false, false>;

// Deprecated spellings (same section for discoverability).

/**
 * @deprecated Use `strided_owning_dataset<DataT, IdxT>` directly.
 *             `LayoutPolicy` / `ContainerPolicy` are legacy parameters and ignored.
 */
template <typename DataT,
          typename IdxT,
          typename LayoutPolicy    = void,
          typename ContainerPolicy = void>
using owning_dataset [[deprecated("Use strided_owning_dataset<DataT, IdxT> directly.")]] =
  strided_owning_dataset<DataT, IdxT>;

/**
 * @deprecated Use `strided_dataset_view<DataT, IdxT>` directly.
 */
template <typename DataT, typename IdxT>
using non_owning_dataset [[deprecated("Use strided_dataset_view<DataT, IdxT> directly.")]] =
  strided_dataset_view<DataT, IdxT>;

/**
 * @deprecated Legacy public spelling; same type as `non_owning_dataset` / `strided_dataset_view`.
 */
template <typename DataT, typename IdxT>
using strided_dataset [[deprecated("Use strided_dataset_view<DataT, IdxT> directly.")]] =
  strided_dataset_view<DataT, IdxT>;

template <typename DatasetT>
struct is_strided_dataset : std::false_type {};

template <typename DataT, typename IdxT>
struct is_strided_dataset<strided_dataset_view<DataT, IdxT>> : std::true_type {};

template <typename DataT, typename IdxT>
struct is_strided_dataset<strided_owning_dataset<DataT, IdxT>> : std::true_type {};

template <typename DatasetT>
[[deprecated(
  "Prefer is_padded_dataset_v where applicable; strided layout dataset/view types are "
  "deprecated.")]]
inline constexpr bool is_strided_dataset_v = is_strided_dataset<DatasetT>::value;

template <typename DatasetT>
struct is_padded_dataset : std::false_type {};

template <typename DataT, typename IdxT>
struct is_padded_dataset<padded_dataset<DataT, IdxT>> : std::true_type {};

template <typename DataT, typename IdxT>
struct is_padded_dataset<padded_dataset_view<DataT, IdxT>> : std::true_type {};

template <typename DatasetT>
inline constexpr bool is_padded_dataset_v = is_padded_dataset<DatasetT>::value;

template <typename DatasetT>
struct is_vpq_dataset : std::false_type {};

template <typename DataT, typename IdxT>
struct is_vpq_dataset<vpq_dataset<DataT, IdxT>> : std::true_type {};

template <typename DatasetT>
inline constexpr bool is_vpq_dataset_v = is_vpq_dataset<DatasetT>::value;

// -----------------------------------------------------------------------------
// CAGRA row width in elements (same for make_padded_dataset* and index layout checks).
// -----------------------------------------------------------------------------

/**
 * @brief Required row width in elements for CAGRA: minimum leading dimension (LDA) per row for the
 *        default per-row byte alignment (16 bytes, combined with `sizeof` element type), given
 *        `logical_columns` feature columns.
 */
[[nodiscard]] inline uint32_t cagra_required_row_width(uint32_t logical_columns,
                                                       std::size_t sizeof_value,
                                                       uint32_t align_bytes = 16)
{
  return static_cast<uint32_t>(
    raft::round_up_safe<std::size_t>(static_cast<std::size_t>(logical_columns) * sizeof_value,
                                     std::lcm(align_bytes, static_cast<uint32_t>(sizeof_value))) /
    sizeof_value);
}

template <typename ValueT>
[[nodiscard]] inline uint32_t cagra_required_row_width(uint32_t logical_columns,
                                                       uint32_t align_bytes = 16)
{
  return cagra_required_row_width(logical_columns, sizeof(ValueT), align_bytes);
}

/** Actual row width in elements (leading dimension) of a 2D `device_matrix_view`. */
template <typename T, typename I, typename L>
[[nodiscard]] inline uint32_t device_matrix_actual_row_width(raft::device_matrix_view<T, I, L> m)
{
  return m.stride(0) > 0 ? static_cast<uint32_t>(m.stride(0)) : static_cast<uint32_t>(m.extent(1));
}

/**
 * @brief True if the matrix's row width in elements matches `cagra_required_row_width` for
 *        `m.extent(1)` and element type `T` (CAGRA row layout is satisfied for this view).
 */
template <typename T, typename I, typename L>
[[nodiscard]] inline bool device_matrix_row_width_matches_cagra_required(
  raft::device_matrix_view<T, I, L> m, uint32_t align_bytes = 16)
{
  using value_type = std::remove_const_t<T>;
  const uint32_t need =
    cagra_required_row_width<value_type>(static_cast<uint32_t>(m.extent(1)), align_bytes);
  const uint32_t actual = device_matrix_actual_row_width(m);
  return actual == need;
}

template <typename DataT, typename IdxT>
[[nodiscard]] inline auto wrap_any_owning(std::unique_ptr<padded_dataset<DataT, IdxT>>&& p)
  -> std::unique_ptr<any_owning_dataset<IdxT>>
{
  return std::make_unique<any_owning_dataset<IdxT>>(std::move(*p));
}

/**
 * @deprecated Prefer `make_padded_dataset` / `make_padded_dataset_view` for CAGRA layout.
 */
template <typename SrcT>
[[deprecated("Prefer make_padded_dataset / make_padded_dataset_view for CAGRA-compatible layout.")]]
auto make_strided_dataset(const raft::resources& res, const SrcT& src, uint32_t required_stride)
  -> std::variant<
    std::unique_ptr<strided_owning_dataset<typename SrcT::value_type, typename SrcT::index_type>>,
    strided_dataset_view<typename SrcT::value_type, typename SrcT::index_type>>
{
  using extents_type = typename SrcT::extents_type;
  using value_type   = typename SrcT::value_type;
  using index_type   = typename SrcT::index_type;
  using layout_type  = typename SrcT::layout_type;
  static_assert(extents_type::rank() == 2, "The input must be a matrix.");
  static_assert(std::is_same_v<layout_type, raft::layout_right> ||
                  std::is_same_v<layout_type, raft::layout_right_padded<value_type>> ||
                  std::is_same_v<layout_type, raft::layout_stride>,
                "The input must be row-major");
  RAFT_EXPECTS(src.extent(1) <= required_stride,
               "The input row length must be not larger than the desired stride.");
  cudaPointerAttributes ptr_attrs;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&ptr_attrs, src.data_handle()));
  auto* device_ptr             = reinterpret_cast<value_type*>(ptr_attrs.devicePointer);
  const uint32_t src_stride    = src.stride(0) > 0 ? src.stride(0) : src.extent(1);
  const bool device_accessible = device_ptr != nullptr;
  const bool row_major         = src.stride(1) <= 1;
  const bool stride_matches    = required_stride == src_stride;

  if (device_accessible && row_major && stride_matches) {
    return strided_dataset_view<value_type, index_type>(
      raft::make_device_strided_matrix_view<const value_type, index_type>(
        device_ptr, src.extent(0), src.extent(1), required_stride));
  }
  auto out_layout = raft::make_strided_layout(
    raft::matrix_extent<index_type>{src.extent(0), src.extent(1)},
    cuda::std::array<index_type, 2>{static_cast<index_type>(required_stride), 1});
  using strided_mat = raft::device_matrix<value_type, index_type, raft::layout_stride>;
  typename strided_mat::container_policy_type cp{};
  strided_mat storage(res, out_layout, cp);

  RAFT_CUDA_TRY(cudaMemsetAsync(storage.data_handle(),
                                0,
                                storage.size() * sizeof(value_type),
                                raft::resource::get_cuda_stream(res)));
  raft::copy_matrix(storage.data_handle(),
                    required_stride,
                    src.data_handle(),
                    src_stride,
                    src.extent(1),
                    src.extent(0),
                    raft::resource::get_cuda_stream(res));

  return std::make_unique<strided_owning_dataset<value_type, index_type>>(std::move(storage),
                                                                          out_layout);
}

template <typename DataT, typename IdxT, typename LayoutPolicy, typename ContainerPolicy>
[[deprecated("Prefer make_padded_dataset / make_padded_dataset_view for CAGRA-compatible layout.")]]
auto make_strided_dataset(
  const raft::resources& res,
  raft::mdarray<DataT, raft::matrix_extent<IdxT>, LayoutPolicy, ContainerPolicy>&& src,
  uint32_t required_stride) -> std::unique_ptr<strided_owning_dataset<DataT, IdxT>>
{
  using value_type            = DataT;
  using index_type            = IdxT;
  using layout_type           = LayoutPolicy;
  using container_policy_type = ContainerPolicy;
  static_assert(std::is_same_v<layout_type, raft::layout_right> ||
                  std::is_same_v<layout_type, raft::layout_right_padded<value_type>> ||
                  std::is_same_v<layout_type, raft::layout_stride>,
                "The input must be row-major");
  RAFT_EXPECTS(src.extent(1) <= required_stride,
               "The input row length must be not larger than the desired stride.");
  const uint32_t src_stride = src.stride(0) > 0 ? src.stride(0) : src.extent(1);
  const bool stride_matches = required_stride == src_stride;

  auto out_layout =
    raft::make_strided_layout(src.extents(), cuda::std::array<index_type, 2>{required_stride, 1});

  using out_mdarray_type          = raft::device_matrix<value_type, index_type>;
  using out_layout_type           = typename out_mdarray_type::layout_type;
  using out_container_policy_type = typename out_mdarray_type::container_policy_type;
  using out_owning_type           = strided_owning_dataset<value_type, index_type>;

  if constexpr (std::is_same_v<layout_type, out_layout_type> &&
                std::is_same_v<container_policy_type, out_container_policy_type>) {
    if (stride_matches) { return std::make_unique<out_owning_type>(std::move(src), out_layout); }
  }
  using strided_mat = raft::device_matrix<value_type, index_type, raft::layout_stride>;
  typename strided_mat::container_policy_type cp{};
  strided_mat storage(res, out_layout, cp);

  RAFT_CUDA_TRY(cudaMemsetAsync(storage.data_handle(),
                                0,
                                storage.size() * sizeof(value_type),
                                raft::resource::get_cuda_stream(res)));
  raft::copy_matrix(storage.data_handle(),
                    required_stride,
                    src.data_handle(),
                    src_stride,
                    src.extent(1),
                    src.extent(0),
                    raft::resource::get_cuda_stream(res));

  return std::make_unique<out_owning_type>(std::move(storage), out_layout);
}

template <typename SrcT>
[[deprecated("Prefer make_padded_dataset / make_padded_dataset_view for CAGRA-compatible layout.")]]
auto make_aligned_dataset(const raft::resources& res, SrcT src, uint32_t align_bytes = 16)
  -> decltype(make_strided_dataset(std::declval<raft::resources const&>(),
                                   std::declval<SrcT>(),
                                   std::declval<uint32_t>()))
{
  using source_type = std::remove_cv_t<std::remove_reference_t<SrcT>>;
  using value_type  = typename source_type::value_type;
  uint32_t required_stride =
    cagra_required_row_width<value_type>(static_cast<uint32_t>(src.extent(1)), align_bytes);
  return make_strided_dataset(res, std::forward<SrcT>(src), required_stride);
}

template <typename SrcT>
auto make_padded_dataset_view(const raft::resources& res,
                              SrcT const& src,
                              uint32_t align_bytes = 16)
  -> device_padded_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  uint32_t required_stride =
    cagra_required_row_width<value_type>(static_cast<uint32_t>(src.extent(1)), align_bytes);
  uint32_t src_stride = src.stride(0) > 0 ? static_cast<uint32_t>(src.stride(0)) : src.extent(1);
  cudaPointerAttributes ptr_attrs;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&ptr_attrs, src.data_handle()));
  auto* device_ptr = reinterpret_cast<value_type*>(ptr_attrs.devicePointer);
  RAFT_EXPECTS(device_ptr != nullptr,
               "make_padded_dataset_view: source must be device-accessible. "
               "Use make_padded_dataset() to get an owning copy.");
  RAFT_EXPECTS(src_stride == required_stride,
               "make_padded_dataset_view: stride is incorrect (required stride for alignment). "
               "Use make_padded_dataset() to get an owning padded copy.");
  auto v =
    raft::make_device_matrix_view(device_ptr, src.extent(0), static_cast<index_type>(src_stride));
  return device_padded_dataset_view<value_type, index_type>(v, src.extent(1));
}

template <typename SrcT>
auto make_padded_dataset(const raft::resources& res, SrcT const& src, uint32_t align_bytes = 16)
  -> std::unique_ptr<device_padded_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  uint32_t required_stride =
    cagra_required_row_width<value_type>(static_cast<uint32_t>(src.extent(1)), align_bytes);
  uint32_t src_stride = src.stride(0) > 0 ? static_cast<uint32_t>(src.stride(0)) : src.extent(1);
  cudaPointerAttributes ptr_attrs;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&ptr_attrs, src.data_handle()));
  bool const device_src =
    (ptr_attrs.type == cudaMemoryTypeDevice) || (ptr_attrs.type == cudaMemoryTypeManaged);
  if (device_src && src_stride == required_stride) {
    RAFT_EXPECTS(false,
                 "make_padded_dataset: source is device and stride is already correct. "
                 "Use make_padded_dataset_view() to get a view instead.");
  }
  RAFT_EXPECTS(src.extent(1) <= required_stride,
               "Source row length must not exceed required stride.");
  auto out_array =
    raft::make_device_matrix<value_type, index_type>(res, src.extent(0), required_stride);
  RAFT_CUDA_TRY(cudaMemsetAsync(out_array.data_handle(),
                                0,
                                out_array.size() * sizeof(value_type),
                                raft::resource::get_cuda_stream(res)));
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(out_array.data_handle(),
                                  sizeof(value_type) * required_stride,
                                  src.data_handle(),
                                  sizeof(value_type) * src_stride,
                                  sizeof(value_type) * src.extent(1),
                                  src.extent(0),
                                  cudaMemcpyDefault,
                                  raft::resource::get_cuda_stream(res)));
  return std::make_unique<device_padded_dataset<value_type, index_type>>(
    std::move(out_array), static_cast<uint32_t>(src.extent(1)));
}

namespace filtering {

/**
 * @defgroup neighbors_filtering Filtering for ANN Types
 * @{
 */

enum class FilterType { None, Bitmap, Bitset };

struct base_filter {
  ~base_filter()                             = default;
  virtual FilterType get_filter_type() const = 0;
};

/* A filter that filters nothing. This is the default behavior. */
struct none_sample_filter : public base_filter {
  /** \cond */
  constexpr __forceinline__ _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the current inverted list index
    const uint32_t cluster_ix,
    // the index of the current sample inside the current inverted list
    const uint32_t sample_ix) const;

  constexpr __forceinline__ _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the index of the current sample
    const uint32_t sample_ix) const;
  /** \endcond */
  FilterType get_filter_type() const override { return FilterType::None; }
};

/**
 * @brief Filter used to convert the cluster index and sample index
 * of an IVF search into a sample index. This can be used as an
 * intermediate filter.
 *
 * @tparam index_t Indexing type
 * @tparam filter_t
 */
template <typename index_t, typename filter_t>
struct ivf_to_sample_filter : public base_filter {
  const index_t* const* inds_ptrs_;
  const filter_t next_filter_;

  _RAFT_HOST_DEVICE ivf_to_sample_filter(const index_t* const* inds_ptrs,
                                         const filter_t next_filter);

  /** \cond */
  /** If the original filter takes three arguments, then don't modify the arguments.
   * If the original filter takes two arguments, then we are using `inds_ptr_` to obtain the sample
   * index.
   */
  inline _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the current inverted list index
    const uint32_t cluster_ix,
    // the index of the current sample inside the current inverted list
    const uint32_t sample_ix) const;

  FilterType get_filter_type() const override { return next_filter_.get_filter_type(); }
  /** \endcond */
};

/**
 * @brief Filter an index with a bitmap
 *
 * @tparam bitmap_t Data type of the bitmap
 * @tparam index_t Indexing type
 */
template <typename bitmap_t, typename index_t>
struct bitmap_filter : public base_filter {
  using view_t = cuvs::core::bitmap_view<bitmap_t, index_t>;

  // View of the bitset to use as a filter
  const view_t bitmap_view_;

  bitmap_filter(const view_t bitmap_for_filtering);
  /** \cond */
  inline _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the index of the current sample
    const uint32_t sample_ix) const;
  /** \endcond */

  FilterType get_filter_type() const override { return FilterType::Bitmap; }

  view_t view() const { return bitmap_view_; }

  template <typename csr_matrix_t>
  void to_csr(raft::resources const& handle, csr_matrix_t& csr);
};

/**
 * @brief Filter an index with a bitset
 *
 * @tparam bitset_t Data type of the bitset
 * @tparam index_t Indexing type
 */
template <typename bitset_t, typename index_t>
struct bitset_filter : public base_filter {
  using view_t = cuvs::core::bitset_view<bitset_t, index_t>;

  // View of the bitset to use as a filter
  const view_t bitset_view_;

  /** \cond */
  _RAFT_HOST_DEVICE bitset_filter(const view_t bitset_for_filtering);
  constexpr __forceinline__ _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the index of the current sample
    const uint32_t sample_ix) const;
  /** \endcond */

  FilterType get_filter_type() const override { return FilterType::Bitset; }

  view_t view() const { return bitset_view_; }

  template <typename csr_matrix_t>
  void to_csr(raft::resources const& handle, csr_matrix_t& csr);
};

/** @} */  // end group neighbors_filtering

/**
 * If the filtering depends on the index of a sample, then the following
 * filter template can be used:
 *
 * template <typename IdxT>
 * struct index_ivf_sample_filter {
 *   using index_type = IdxT;
 *
 *   const index_type* const* inds_ptr = nullptr;
 *
 *   index_ivf_sample_filter() {}
 *   index_ivf_sample_filter(const index_type* const* _inds_ptr)
 *       : inds_ptr{_inds_ptr} {}
 *   index_ivf_sample_filter(const index_ivf_sample_filter&) = default;
 *   index_ivf_sample_filter(index_ivf_sample_filter&&) = default;
 *   index_ivf_sample_filter& operator=(const index_ivf_sample_filter&) = default;
 *   index_ivf_sample_filter& operator=(index_ivf_sample_filter&&) = default;
 *
 *   inline _RAFT_HOST_DEVICE bool operator()(
 *       const uint32_t query_ix,
 *       const uint32_t cluster_ix,
 *       const uint32_t sample_ix) const {
 *     index_type database_idx = inds_ptr[cluster_ix][sample_ix];
 *
 *     // return true or false, depending on the database_idx
 *     return true;
 *   }
 * };
 *
 * Initialize it as:
 *   using filter_type = index_ivf_sample_filter<idx_t>;
 *   filter_type filter(cuvs_ivfpq_index.inds_ptrs().data_handle());
 *
 * Use it as:
 *   cuvs::neighbors::ivf_pq::search_with_filtering<data_t, idx_t, filter_type>(
 *     ...regular parameters here...,
 *     filter
 *   );
 *
 * Another example would be the following filter that greenlights samples according
 * to a contiguous bit mask vector.
 *
 * template <typename IdxT>
 * struct bitmask_ivf_sample_filter {
 *   using index_type = IdxT;
 *
 *   const index_type* const* inds_ptr = nullptr;
 *   const uint64_t* const bit_mask_ptr = nullptr;
 *   const int64_t bit_mask_stride_64 = 0;
 *
 *   bitmask_ivf_sample_filter() {}
 *   bitmask_ivf_sample_filter(
 *       const index_type* const* _inds_ptr,
 *       const uint64_t* const _bit_mask_ptr,
 *       const int64_t _bit_mask_stride_64)
 *       : inds_ptr{_inds_ptr},
 *         bit_mask_ptr{_bit_mask_ptr},
 *         bit_mask_stride_64{_bit_mask_stride_64} {}
 *   bitmask_ivf_sample_filter(const bitmask_ivf_sample_filter&) = default;
 *   bitmask_ivf_sample_filter(bitmask_ivf_sample_filter&&) = default;
 *   bitmask_ivf_sample_filter& operator=(const bitmask_ivf_sample_filter&) = default;
 *   bitmask_ivf_sample_filter& operator=(bitmask_ivf_sample_filter&&) = default;
 *
 *   inline _RAFT_HOST_DEVICE bool operator()(
 *       const uint32_t query_ix,
 *       const uint32_t cluster_ix,
 *       const uint32_t sample_ix) const {
 *     const index_type database_idx = inds_ptr[cluster_ix][sample_ix];
 *     const uint64_t bit_mask_element =
 *         bit_mask_ptr[query_ix * bit_mask_stride_64 + database_idx / 64];
 *     const uint64_t masked_bool =
 *         bit_mask_element & (1ULL << (uint64_t)(database_idx % 64));
 *     const bool is_bit_set = (masked_bool != 0);
 *
 *     return is_bit_set;
 *   }
 * };
 */
}  // namespace filtering

namespace ivf {

/**
 * Default value filled in the `indices` array.
 * One may encounter it trying to access a record within a list that is outside of the
 * `size` bound or whenever the list is allocated but not filled-in yet.
 */
template <typename IdxT>
constexpr static IdxT kInvalidRecord =
  (std::is_signed_v<IdxT> ? IdxT{0} : std::numeric_limits<IdxT>::max()) - 1;

/**
 * Abstract base class for IVF list data.
 * This allows polymorphic access to list data regardless of the underlying layout.
 *
 * @tparam ValueT The data element type (e.g., uint8_t for PQ codes, float for raw vectors)
 * @tparam IdxT The index type for source indices
 * @tparam SizeT The size type
 *
 * TODO: Make this struct internal (tracking issue: https://github.com/rapidsai/cuvs/issues/1726)
 */
template <typename ValueT, typename IdxT, typename SizeT = uint32_t>
struct list_base {
  using value_type = ValueT;
  using index_type = IdxT;
  using size_type  = SizeT;

  virtual ~list_base() = default;

  /** Get the raw data pointer. */
  virtual value_type* data_ptr() noexcept             = 0;
  virtual const value_type* data_ptr() const noexcept = 0;

  /** Get the indices pointer. */
  virtual index_type* indices_ptr() noexcept             = 0;
  virtual const index_type* indices_ptr() const noexcept = 0;

  /** Get the current size (number of records). */
  virtual size_type get_size() const noexcept = 0;

  /** Set the current size (number of records). */
  virtual void set_size(size_type new_size) noexcept = 0;

  /** Get the total size of the data array in bytes. */
  virtual size_t data_byte_size() const noexcept = 0;

  /** Get the capacity (number of indices that can be stored). */
  virtual size_type indices_capacity() const noexcept = 0;
};

/** The data for a single IVF list. */
template <template <typename, typename...> typename SpecT,
          typename SizeT,
          typename... SpecExtraArgs>
struct list : public list_base<typename SpecT<SizeT, SpecExtraArgs...>::value_type,
                               typename SpecT<SizeT, SpecExtraArgs...>::index_type,
                               SizeT> {
  using size_type    = SizeT;
  using spec_type    = SpecT<size_type, SpecExtraArgs...>;
  using value_type   = typename spec_type::value_type;
  using index_type   = typename spec_type::index_type;
  using list_extents = typename spec_type::list_extents;

  /** Possibly encoded data; it's layout is defined by `SpecT`. */
  raft::device_mdarray<value_type, list_extents, raft::row_major> data;
  /** Source indices. */
  raft::device_mdarray<index_type, raft::extent_1d<size_type>, raft::row_major> indices;
  /** The actual size of the content. */
  std::atomic<size_type> size;

  /** Allocate a new list capable of holding at least `n_rows` data records and indices. */
  list(raft::resources const& res, const spec_type& spec, size_type n_rows);

  value_type* data_ptr() noexcept override { return data.data_handle(); }
  const value_type* data_ptr() const noexcept override { return data.data_handle(); }

  index_type* indices_ptr() noexcept override { return indices.data_handle(); }
  const index_type* indices_ptr() const noexcept override { return indices.data_handle(); }

  size_type get_size() const noexcept override { return size.load(); }
  void set_size(size_type new_size) noexcept override { size.store(new_size); }

  size_t data_byte_size() const noexcept override { return data.size() * sizeof(value_type); }
  size_type indices_capacity() const noexcept override { return indices.extent(0); }
};

template <typename ListT, class T = void>
struct enable_if_valid_list {};

template <class T,
          template <typename, typename...> typename SpecT,
          typename SizeT,
          typename... SpecExtraArgs>
struct enable_if_valid_list<list<SpecT, SizeT, SpecExtraArgs...>, T> {
  using type = T;
};

/**
 * Designed after `std::enable_if_t`, this trait is helpful in the instance resolution;
 * plug this in the return type of a function that has an instance of `ivf::list` as
 * a template parameter.
 */
template <typename ListT, class T = void>
using enable_if_valid_list_t = typename enable_if_valid_list<ListT, T>::type;

/**
 * Resize a list by the given id, so that it can contain the given number of records;
 * copy the data if necessary.
 *
 * @note This is an internal function that requires the concrete list type.
 *       For IVF-PQ indexes, prefer using the helper functions in
 *       `cuvs::neighbors::ivf_pq::helpers::resize_list` which handle type casting internally.
 */
template <typename ListT>
void resize_list(raft::resources const& res,
                 std::shared_ptr<ListT>& orig_list,  // NOLINT
                 const typename ListT::spec_type& spec,
                 typename ListT::size_type new_used_size,
                 typename ListT::size_type old_used_size);

/**
 * Serialize a list to an output stream.
 *
 * @note This function requires the concrete list type (not the base class) because:
 *       1. It needs access to the spec_type to determine the data layout for serialization
 *       2. The serialized format depends on the spec's make_list_extents() method
 *       When calling from code that only has a base class pointer, use std::static_pointer_cast
 *       to obtain the typed pointer first.
 */
template <typename ListT>
enable_if_valid_list_t<ListT> serialize_list(
  const raft::resources& handle,
  std::ostream& os,
  const ListT& ld,
  const typename ListT::spec_type& store_spec,
  std::optional<typename ListT::size_type> size_override = std::nullopt);

template <typename ListT>
enable_if_valid_list_t<ListT> serialize_list(
  const raft::resources& handle,
  std::ostream& os,
  const std::shared_ptr<ListT>& ld,
  const typename ListT::spec_type& store_spec,
  std::optional<typename ListT::size_type> size_override = std::nullopt);

template <typename ListT>
enable_if_valid_list_t<ListT> deserialize_list(const raft::resources& handle,
                                               std::istream& is,
                                               std::shared_ptr<ListT>& ld,
                                               const typename ListT::spec_type& store_spec,
                                               const typename ListT::spec_type& device_spec);
}  // namespace ivf

using namespace raft;

template <typename AnnIndexType, typename T, typename IdxT>
struct iface {
  iface() : cagra_owned_dataset_(nullptr), mutex_(std::make_shared<std::mutex>()) {}

  const IdxT size() const { return index_.value().size(); }

  std::optional<AnnIndexType> index_;
  /** Used by CAGRA when built from host: holds device copy so index dataset view stays valid. */
  std::optional<raft::device_matrix<T, int64_t, raft::row_major>> cagra_build_dataset_;
  /** Used by CAGRA when deserializing an index that contains a dataset; keeps it alive for the
   * view. */
  std::unique_ptr<cuvs::neighbors::any_owning_dataset<int64_t>> cagra_owned_dataset_;
  std::shared_ptr<std::mutex> mutex_;
};

template <typename AnnIndexType, typename T, typename IdxT, typename Accessor>
void build(const raft::resources& handle,
           cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
           const cuvs::neighbors::index_params* index_params,
           raft::mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> index_dataset);

template <typename AnnIndexType, typename T, typename IdxT, typename Accessor1, typename Accessor2>
void extend(
  const raft::resources& handle,
  cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
  raft::mdspan<const T, matrix_extent<int64_t>, row_major, Accessor1> new_vectors,
  std::optional<raft::mdspan<const IdxT, vector_extent<int64_t>, layout_c_contiguous, Accessor2>>
    new_indices);

template <typename AnnIndexType, typename T, typename IdxT, typename searchIdxT>
void search(const raft::resources& handle,
            const cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
            const cuvs::neighbors::search_params* search_params,
            raft::device_matrix_view<const T, int64_t, row_major> h_queries,
            raft::device_matrix_view<searchIdxT, int64_t, row_major> d_neighbors,
            raft::device_matrix_view<float, int64_t, row_major> d_distances);

template <typename AnnIndexType, typename T, typename IdxT>
void serialize(const raft::resources& handle,
               const cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
               std::ostream& os);

template <typename AnnIndexType, typename T, typename IdxT>
void deserialize(const raft::resources& handle,
                 cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
                 std::istream& is);

template <typename AnnIndexType, typename T, typename IdxT>
void deserialize(const raft::resources& handle,
                 cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
                 const std::string& filename);

/// \defgroup mg_cpp_index_params ANN MG index build parameters

/** Distribution mode */
/// \ingroup mg_cpp_index_params
enum distribution_mode {
  /** Index is replicated on each device, favors throughput */
  REPLICATED,
  /** Index is split on several devices, favors scaling */
  SHARDED
};

/// \defgroup mg_cpp_search_params ANN MG search parameters

/** Search mode when using a replicated index */
/// \ingroup mg_cpp_search_params
enum replicated_search_mode {
  /** Search queries are split to maintain equal load on GPUs */
  LOAD_BALANCER,
  /** Each search query is processed by a single GPU in a round-robin fashion */
  ROUND_ROBIN
};

/** Merge mode when using a sharded index */
/// \ingroup mg_cpp_search_params
enum sharded_merge_mode {
  /** Search batches are merged on the root rank */
  MERGE_ON_ROOT_RANK,
  /** Search batches are merged in a tree reduction fashion */
  TREE_MERGE
};

/** Build parameters */
/// \ingroup mg_cpp_index_params
template <typename Upstream>
struct mg_index_params : public Upstream {
  mg_index_params() : mode(SHARDED) {}

  mg_index_params(const Upstream& sp) : Upstream(sp), mode(SHARDED) {}

  /** Distribution mode */
  cuvs::neighbors::distribution_mode mode = SHARDED;
};

/** Search parameters */
/// \ingroup mg_cpp_search_params
template <typename Upstream>
struct mg_search_params : public Upstream {
  mg_search_params() : search_mode(LOAD_BALANCER), merge_mode(TREE_MERGE) {}

  mg_search_params(const Upstream& sp)
    : Upstream(sp), search_mode(LOAD_BALANCER), merge_mode(TREE_MERGE)
  {
  }

  /** Replicated search mode */
  cuvs::neighbors::replicated_search_mode search_mode = LOAD_BALANCER;
  /** Sharded merge mode */
  cuvs::neighbors::sharded_merge_mode merge_mode = TREE_MERGE;
  /** Number of rows per batch */
  int64_t n_rows_per_batch = 1 << 20;
};

template <typename AnnIndexType, typename T, typename IdxT>
struct mg_index {
  mg_index(const raft::resources& clique, distribution_mode mode);
  mg_index(const raft::resources& clique, const std::string& filename);

  mg_index(const mg_index&)                    = delete;
  mg_index(mg_index&&)                         = default;
  auto operator=(const mg_index&) -> mg_index& = delete;
  auto operator=(mg_index&&) -> mg_index&      = default;

  distribution_mode mode_;
  int num_ranks_;
  std::vector<iface<AnnIndexType, T, IdxT>> ann_interfaces_;

  // for load balancing mechanism
  std::shared_ptr<std::atomic<int64_t>> round_robin_counter_;
};

}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
