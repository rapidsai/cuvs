/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>

#include <raft/core/coo_matrix.hpp>
#include <raft/core/csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/util/integer_utils.hpp>

#include <concepts>
#include <type_traits>
#include <utility>
#include <variant>

namespace CUVS_EXPORT cuvs {

/************************************************************************************************
 * The core dataset type
 ************************************************************************************************/

/**
 * A spec is compressing if it defines a dictionary: the codebooks, the quantization range, or
 * whatever else is needed to interpret the data. Plain specs define none and the dataset
 * substitutes `std::monostate`, so "there is no dictionary" and "the dictionary slot is empty" are
 * the same vocabulary type.
 */
template <typename SpecT>
concept compressed_dataset_spec = requires { typename SpecT::dictionary_type; };

/**
 * A non-owning view of a dataset: a view of the data and, for a compressed dataset, a view of the
 * dictionary that gives it meaning. Cheap to copy and to pass by value, like an mdspan.
 */
template <typename T, typename IdxT, typename SpecT>
struct dataset_view {
  using spec_type = typename SpecT::template apply<T, IdxT>;
  /* The read-only flavor, in which each view type collapses onto its const counterpart. */
  static constexpr bool is_const_view = std::is_const_v<T>;

  using value_type = typename spec_type::value_type;
  using index_type = typename spec_type::index_type;
  /**
    What one element of the data slot is
      - for compressed datasets it is the raw data element type; often is uint8_t but may be
        a larger type to enforce better alignment.
      - for uncompressed datasets it is the value type of the dataset.
   */
  using code_type = typename decltype([] {
    if constexpr (requires { typename spec_type::code_type; }) {
      return std::type_identity<typename spec_type::code_type>{};
    } else {
      return std::type_identity<value_type>{};
    }
  }())::type;

  using data_type            = typename spec_type::data_type;
  using const_data_view_type = typename spec_type::const_data_view_type;
  using data_view_type =
    std::conditional_t<is_const_view, const_data_view_type, typename spec_type::data_view_type>;

  /**
    Optional slot that contains everything to give meaning to the encoded data;
    `std::monostate` for an uncompressed dataset.
   */
  using dictionary_type = typename decltype([] {
    if constexpr (requires { typename spec_type::dictionary_type; }) {
      return std::type_identity<typename spec_type::dictionary_type>{};
    } else {
      return std::type_identity<std::monostate>{};
    }
  }())::type;
  // Optional, so that the specs don't have to declare it; a spec that declares neither view uses
  // the dictionary itself, and one that declares only the mutable view uses it for both.
  using const_dictionary_view_type = typename decltype([] {
    if constexpr (requires { typename spec_type::const_dictionary_view_type; }) {
      return std::type_identity<typename spec_type::const_dictionary_view_type>{};
    } else if constexpr (requires { typename spec_type::dictionary_view_type; }) {
      return std::type_identity<typename spec_type::dictionary_view_type>{};
    } else {
      return std::type_identity<dictionary_type>{};
    }
  }())::type;
  // Optional, so that the specs don't have to declare it
  using dictionary_view_type = typename decltype([] {
    if constexpr (is_const_view) {
      return std::type_identity<const_dictionary_view_type>{};
    } else if constexpr (requires { typename spec_type::dictionary_view_type; }) {
      return std::type_identity<typename spec_type::dictionary_view_type>{};
    } else {
      return std::type_identity<dictionary_type>{};
    }
  }())::type;

  static constexpr bool is_compressed = compressed_dataset_spec<spec_type>;

  explicit dataset_view(data_view_type data_view)
    requires(!is_compressed)
    : data_view_{data_view}
  {
  }

  dataset_view(data_view_type data_view, dictionary_view_type dictionary_view)
    requires(is_compressed)
    : data_view_{data_view}, dictionary_view_{dictionary_view}
  {
  }

  /* A mutable view decays to a read-only one, the way an mdspan does.

  NOTE: a constructor template, because in the mutable flavor the parameter would spell the very
  same type; a non-template would be a copy constructor there, and declaring one - even a
  constrained-away one - suppresses the implicitly declared copy constructor.

  NOTE: the dictionary is assigned in the body rather than in the init-list, because the accessor it
  comes from is constrained on `is_compressed`.
  */
  template <typename OtherT>
    requires(is_const_view && std::same_as<OtherT, std::remove_const_t<T>>)
  dataset_view(const dataset_view<OtherT, IdxT, SpecT>& other) noexcept
    : data_view_{other.data_const_view()}
  {
    if constexpr (is_compressed) { dictionary_view_ = other.const_dictionary_view(); }
  }

  [[nodiscard]] auto n_rows() const noexcept -> index_type
  {
    return spec_type::get_n_rows(data_view_);
  }
  // NOTE: a compressed dataset may not be able to tell its dimension from the encoded data alone
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    return spec_type::get_dim(data_view_, dictionary_view_);
  }

  /*
   @return A non-owning view of the data, such as mdspan.
  */
  [[nodiscard]] auto data_view() const noexcept -> data_view_type { return data_view_; }
  /*
   NOTE: giving up write access is not a plain cast for most view types - a mutable mdspan, for
   one, has to be rebuilt over a const accessor - hence the spec does the conversion.

   @return A read-only view of the data.
  */
  [[nodiscard]] auto data_const_view() const noexcept -> const_data_view_type
  {
    if constexpr (is_const_view) {
      return data_view_;
    } else {
      return spec_type::get_data_const_view(data_view_);
    }
  }

  /*
   @return A non-owning view of the dictionary, such as the two codebooks of a VPQ dataset.
  */
  [[nodiscard]] auto dictionary_view() const noexcept -> dictionary_view_type
    requires(is_compressed)
  {
    return dictionary_view_;
  }
  /*
   @return A read-only view of the dictionary.
  */
  [[nodiscard]] auto const_dictionary_view() const noexcept -> const_dictionary_view_type
    requires(is_compressed)
  {
    if constexpr (is_const_view) {
      return dictionary_view_;
    } else {
      return spec_type::get_dictionary_const_view(dictionary_view_);
    }
  }

 private:
  data_view_type data_view_;
  [[no_unique_address]] dictionary_view_type dictionary_view_;
};

/**
 * An owning dataset: the data and, for a compressed dataset, the dictionary that gives it meaning.
 * Ownership is unique, so the type is move-only whenever the underlying storage is.
 */
template <typename T, typename IdxT, typename SpecT>
struct dataset {
  /* NOTE: element type for an owning dataset cannot be const;
     the constness of the dataset itself selects the flavor of the view it hands out.
  */
  static_assert(!std::is_const_v<T>, "Element type for an owning dataset must not be const.");

  using view_type       = dataset_view<T, IdxT, SpecT>;
  using const_view_type = dataset_view<const T, IdxT, SpecT>;

  using spec_type  = typename view_type::spec_type;
  using value_type = typename view_type::value_type;
  using index_type = typename view_type::index_type;
  using code_type  = typename view_type::code_type;

  using data_type            = typename view_type::data_type;
  using data_view_type       = typename view_type::data_view_type;
  using const_data_view_type = typename view_type::const_data_view_type;

  /**
    Optional slot that contains everything to give meaning to the encoded data;
    `std::monostate` for an uncompressed dataset.
   */
  using dictionary_type            = typename view_type::dictionary_type;
  using dictionary_view_type       = typename view_type::dictionary_view_type;
  using const_dictionary_view_type = typename view_type::const_dictionary_view_type;

  static constexpr bool is_compressed = view_type::is_compressed;

  explicit dataset(data_type&& data)
    requires(!is_compressed)
    : data_{std::move(data)}
  {
  }

  dataset(data_type&& data, dictionary_type&& dictionary)
    requires(is_compressed)
    : data_{std::move(data)}, dictionary_{std::move(dictionary)}
  {
  }

  /*
   @return A non-owning view of the whole dataset, data and dictionary alike.
  */
  [[nodiscard]] auto view() noexcept -> view_type
  {
    if constexpr (is_compressed) {
      return view_type{data_view(), dictionary_view()};
    } else {
      return view_type{data_view()};
    }
  }
  /*
   @return A read-only view of the whole dataset; the constness of the dataset selects this flavor.
  */
  [[nodiscard]] auto view() const noexcept -> const_view_type
  {
    if constexpr (is_compressed) {
      return const_view_type{data_const_view(), const_dictionary_view()};
    } else {
      return const_view_type{data_const_view()};
    }
  }

  [[nodiscard]] auto n_rows() const noexcept -> index_type { return spec_type::get_n_rows(data_); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    // NOTE: a compressed dataset may not be able to tell its dimension from the encoded data alone
    return spec_type::get_dim(data_, dictionary_);
  }

  /*
   @return A non-owning view of the data, such as mdspan.
  */
  [[nodiscard]] auto data_view() noexcept -> data_view_type
  {
    return spec_type::get_data_view(data_);
  }
  [[nodiscard]] auto data_const_view() const noexcept -> const_data_view_type
  {
    return spec_type::get_data_const_view(data_);
  }

  /*
   @return A non-owning view of the dictionary, such as the two codebooks of a VPQ dataset.
  */
  [[nodiscard]] auto dictionary_view() const noexcept -> dictionary_view_type
    requires(is_compressed)
  {
    return spec_type::get_dictionary_view(dictionary_);
  }
  /*
   @return A read-only view of the dictionary.
  */
  [[nodiscard]] auto const_dictionary_view() const noexcept -> const_dictionary_view_type
    requires(is_compressed)
  {
    return spec_type::get_dictionary_const_view(dictionary_);
  }

 private:
  data_type data_;
  [[no_unique_address]] dictionary_type dictionary_;
};

/************************************************************************************************
 * Dataset specifications
 ************************************************************************************************/

/**
 * Empty dataset specification: contains no data, but keeps the dimension of the dataset implement
 * the common interface.
 */
struct empty_spec {
  struct empty_dataset_rep {
    uint32_t dim;
  };

  template <typename T, typename IdxT>
  struct apply {
    using data_type      = empty_dataset_rep;
    using data_view_type = empty_dataset_rep;
    /* NOTE: not `const empty_dataset_rep`: a read-only view holds this as a member, and a const
    member would delete its assignment operator.
    */
    using const_data_view_type = empty_dataset_rep;
    using value_type           = std::remove_cv_t<T>;
    using index_type           = std::remove_cv_t<IdxT>;

    [[nodiscard]] static auto get_data_view(data_type& data) noexcept -> data_view_type
    {
      return data;
    }
    /* NOTE: the view of an empty dataset is the storage itself, so one overload serves both the
    owning dataset and the view.
    */
    [[nodiscard]] static auto get_data_const_view(const data_type& data) noexcept
      -> const_data_view_type
    {
      return data;
    }
    /* NOTE: the shape accessors take the data in either form - the storage an owning dataset holds,
    or the view a `dataset_view` holds. A plain spec defines no dictionary either, so it ignores
    whatever the dataset puts in that slot instead.
    */
    [[nodiscard]] static auto get_n_rows(const auto&) noexcept -> index_type { return 0; }
    [[nodiscard]] static auto get_dim(const auto& data, const auto&) noexcept -> uint32_t
    {
      return data.dim;
    }
  };
};

/**
 * Dense plain/padded dataset implemented via raft::mdarray.
 */
template <typename LayoutPolicy, typename ContainerPolicy>
struct mdarray_spec {
  template <typename T, typename IdxT>
  struct apply {
    /* NOTE: index type != extents type
    The index type can vary depending on the use case; we often may want to store indices in 32-bit
    slots to save memory on large datasets. The extents are always fixed to 64-bit signed integers
    to simplify the conversion and avoid integer overflow issues.
    */
    using value_type = std::remove_cv_t<T>;
    using index_type = std::remove_cv_t<IdxT>;
    using data_type =
      raft::mdarray<value_type, raft::matrix_extent<int64_t>, LayoutPolicy, ContainerPolicy>;
    using data_view_type       = typename data_type::view_type;
    using const_data_view_type = typename data_type::const_view_type;

    [[nodiscard]] static auto get_data_view(data_type& data) noexcept -> data_view_type
    {
      return data.view();
    }

    [[nodiscard]] static auto get_data_const_view(const data_type& data) noexcept
      -> const_data_view_type
    {
      return data.view();
    }
    /* A mutable mdspan does not decay to a read-only one on its own; it has to be rebuilt over an
    accessor of `const T`.
    */
    [[nodiscard]] static auto get_data_const_view(data_view_type data) noexcept
      -> const_data_view_type
    {
      return raft::make_const_mdspan(data);
    }
    /* NOTE: the shape accessors take the data in either form - the storage an owning dataset holds,
    or the view a `dataset_view` holds. A plain spec defines no dictionary either, so it ignores
    whatever the dataset puts in that slot instead.
    */
    [[nodiscard]] static auto get_n_rows(const auto& data) noexcept -> index_type
    {
      return static_cast<index_type>(data.extent(0));
    }
    [[nodiscard]] static auto get_dim(const auto& data, const auto&) noexcept -> uint32_t
    {
      return static_cast<uint32_t>(data.extent(1));
    }
  };
};

/**
 * Sparse dataset specification implemented via the raft sparse matrix hierarchy. `SparseLayoutT`
 * selects the representation (`csr_layout`, `coo_layout`).
 *
 * NOTE: only the memory space of the container policy is used here; raft's sparse types bind the
 * policy themselves, separately for the values, the offsets and the indices.
 */
template <typename SparseLayoutT, typename ContainerPolicy>
struct sparse_spec {
  template <typename T, typename IdxT>
  struct apply {
    using value_type = std::remove_cv_t<T>;
    using index_type = std::remove_cv_t<IdxT>;

    static constexpr raft::memory_type mem_type = ContainerPolicy::mem_type;
    static_assert(mem_type == raft::memory_type::host || mem_type == raft::memory_type::device,
                  "raft's sparse hierarchy only provides host and device containers");

    using data_type = std::conditional_t<
      mem_type == raft::memory_type::device,
      typename SparseLayoutT::
        template matrix_type<value_type, index_type, true, raft::device_container_policy>,
      typename SparseLayoutT::
        template matrix_type<value_type, index_type, false, raft::host_container_policy>>;
    using data_view_type = typename data_type::view_type;
    /* NOTE: raft's sparse hierarchy provides no read-only view yet (see the TODO in
    raft/core/sparse_types.hpp); until then, even reading the structure requires a mutable matrix.

    Consequently a read-only `dataset_view<const T, ...>` over a sparse spec is read-only in name
    only: the two view types coincide, so writing through it still compiles. That resolves itself
    once raft grows a const sparse view.
    */
    using const_data_view_type = data_view_type;

    [[nodiscard]] static auto get_data_view(data_type& data) noexcept -> data_view_type
    {
      return data.view();
    }

    [[nodiscard]] static auto get_data_const_view(const data_type& data) noexcept
      -> const_data_view_type
    {
      return const_cast<data_type&>(data).view();
    }
    /* NOTE: there is no read-only view to convert to (see above), so the view passes through. */
    [[nodiscard]] static auto get_data_const_view(data_view_type data) noexcept
      -> const_data_view_type
    {
      return data;
    }
    /* NOTE: raft's sparse hierarchy exposes the structure through a mutable accessor only, so
    reading the shape of either the owning matrix or its view takes a cast.
    */
    [[nodiscard]] static auto get_structure_view(const auto& data) noexcept
    {
      return const_cast<std::remove_cvref_t<decltype(data)>&>(data).structure_view();
    }
    /* NOTE: the shape accessors take the data in either form - the storage an owning dataset holds,
    or the view a `dataset_view` holds. A plain spec defines no dictionary either, so it ignores
    whatever the dataset puts in that slot instead.
    */
    [[nodiscard]] static auto get_n_rows(const auto& data) noexcept -> index_type
    {
      return static_cast<index_type>(get_structure_view(data).get_n_rows());
    }
    [[nodiscard]] static auto get_dim(const auto& data, const auto&) noexcept -> uint32_t
    {
      return static_cast<uint32_t>(get_structure_view(data).get_n_cols());
    }
  };
};

/**
 * CSR representation: one compressed row per vector of the dataset.
 *
 * NOTE: the row offsets are 64-bit for the same reason the dense extents are: the number of
 * non-zeros is not bounded by the index type. The column indices are feature ids, so they are
 * bounded by the dataset dimension rather than by its size.
 */
struct csr_layout {
  template <typename T, typename IdxT, bool IsDevice, template <typename> typename ContainerPolicy>
  using matrix_type = raft::csr_matrix<T, int64_t, uint32_t, uint64_t, IsDevice, ContainerPolicy>;
};

/**
 * COO representation: one (row, column, value) triple per non-zero.
 *
 * NOTE: unlike CSR, the row component is an index into the dataset rather than an offset into the
 * non-zeros, hence the dataset index type.
 */
struct coo_layout {
  template <typename T, typename IdxT, bool IsDevice, template <typename> typename ContainerPolicy>
  using matrix_type = raft::coo_matrix<T, IdxT, uint32_t, uint64_t, IsDevice, ContainerPolicy>;
};

/**
 * VQ+PQ compressed dataset specification, mirroring the VPQ dataset of cuvs/neighbors/common.hpp:
 * the two codebooks form the dictionary, while the data slot keeps the encoded rows - the VQ label
 * in the row prefix, followed by the packed PQ codes.
 *
 * `BookPolicy` is the container policy of the codebooks (elements of `MathT`) and `CodePolicy` the
 * one of the encoded rows (elements of `uint8_t`); a bound policy cannot be rebound to another
 * element type, hence the two parameters.
 */
template <typename MathT, typename BookPolicy, typename CodePolicy>
struct vpq_spec {
  /* The encoded rows are stored like the rows of any other dense dataset; only how a row is read
  and where the dimension comes from are specific to the compression.
  */
  template <typename IdxT>
  using storage_spec =
    typename mdarray_spec<raft::layout_c_contiguous, CodePolicy>::template apply<uint8_t, IdxT>;

  /* The type the codebooks are stored in; the rows themselves are always packed codes. */
  using math_type = MathT;

  /* [vq_n_centers, dim] */
  using vq_book_type =
    raft::mdarray<math_type, raft::matrix_extent<int64_t>, raft::layout_c_contiguous, BookPolicy>;
  /* [n_books, pq_n_centers, pq_len], where `n_books` is 1 for a codebook shared by all the
  subspaces, or `pq_dim` for one codebook per subspace. A row-major 3d array with `n_books == 1`
  is bit-identical to the flat 2d codebook of the current implementation.
  */
  using pq_book_type =
    raft::mdarray<math_type, raft::extent_3d<int64_t>, raft::layout_c_contiguous, BookPolicy>;

  struct dictionary_type {
    vq_book_type vq_code_book;
    pq_book_type pq_code_book;
  };

  struct dictionary_view_type {
    typename vq_book_type::const_view_type vq_code_book;
    typename pq_book_type::const_view_type pq_code_book;

    /* NOTE: the dimension of the dataset is a property of the VQ codebook: the encoded rows tell
    only how many bytes it took to compress a vector, which the padding makes ambiguous.
    */
    [[nodiscard]] auto dim() const noexcept -> uint32_t
    {
      return static_cast<uint32_t>(vq_code_book.extent(1));
    }
    [[nodiscard]] auto vq_n_centers() const noexcept -> uint32_t
    {
      return static_cast<uint32_t>(vq_code_book.extent(0));
    }
    [[nodiscard]] auto n_books() const noexcept -> uint32_t
    {
      return static_cast<uint32_t>(pq_code_book.extent(0));
    }
    [[nodiscard]] auto pq_n_centers() const noexcept -> uint32_t
    {
      return static_cast<uint32_t>(pq_code_book.extent(1));
    }
    [[nodiscard]] auto pq_len() const noexcept -> uint32_t
    {
      return static_cast<uint32_t>(pq_code_book.extent(2));
    }
    [[nodiscard]] auto pq_bits() const noexcept -> uint32_t
    {
      return static_cast<uint32_t>(std::countr_zero(pq_n_centers()));
    }
    [[nodiscard]] auto pq_dim() const noexcept -> uint32_t
    {
      return raft::div_rounding_up_safe(dim(), pq_len());
    }
    /* Whether every subspace has a codebook of its own, as opposed to sharing a single one. */
    [[nodiscard]] auto per_subspace() const noexcept -> bool { return n_books() > 1; }
  };

  template <typename T, typename IdxT>
  struct apply : storage_spec<IdxT> {
    /* The dataset stores codes, but it still represents vectors of `T`. */
    using value_type = std::remove_cv_t<T>;
    using code_type  = uint8_t;
    /* NOTE: re-exported rather than declared here, so that they do not belong to one instantiation
    of `apply`; qualified lookup into `apply` does not reach the enclosing spec on its own.
    */
    using math_type            = vpq_spec::math_type;
    using vq_book_type         = vpq_spec::vq_book_type;
    using pq_book_type         = vpq_spec::pq_book_type;
    using dictionary_type      = vpq_spec::dictionary_type;
    using dictionary_view_type = vpq_spec::dictionary_view_type;

    /* NOTE: the data accessors are inherited; only `get_dim` differs from a dense dataset. Like the
    inherited ones, it reads the dictionary in either form: the codebooks an owning dataset holds,
    or the views a `dataset_view` holds.
    */
    [[nodiscard]] static auto get_dim(const auto&, const auto& dict) noexcept -> uint32_t
    {
      return static_cast<uint32_t>(dict.vq_code_book.extent(1));
    }
    [[nodiscard]] static auto get_dictionary_view(const dictionary_type& dict) noexcept
      -> dictionary_view_type
    {
      return {dict.vq_code_book.view(), dict.pq_code_book.view()};
    }
    /* NOTE: a codebook view is read-only to begin with, so the spec declares no separate const view
    and these overloads exist only to spell the conversion out of either form.
    */
    [[nodiscard]] static auto get_dictionary_const_view(const dictionary_type& dict) noexcept
      -> dictionary_view_type
    {
      return get_dictionary_view(dict);
    }
    [[nodiscard]] static auto get_dictionary_const_view(dictionary_view_type dict) noexcept
      -> dictionary_view_type
    {
      return dict;
    }
    /* The length of an encoded row in bytes, including the inlined VQ label. */
    [[nodiscard]] static auto get_encoded_row_length(const auto& data) noexcept -> uint32_t
    {
      return static_cast<uint32_t>(data.extent(1));
    }
  };
};

/**
 * Scalar-quantized dataset specification: every component of a vector becomes one code, so the
 * dictionary is only the range the codes are mapped back onto and the dimension is still a property
 * of the data.
 */
template <typename MathT, typename CodePolicy>
struct sq_spec {
  /* One code per component, so the codes are laid out exactly like a dense dataset. */
  template <typename IdxT>
  using storage_spec =
    typename mdarray_spec<raft::layout_c_contiguous, CodePolicy>::template apply<int8_t, IdxT>;

  using math_type = MathT;

  /* Nothing to allocate: the whole dictionary is the interval the codes are dequantized into.
  The members are named as in cuvs::preprocessing::quantize::scalar::quantizer.
  */
  struct dictionary_type {
    math_type min_;
    math_type max_;
  };
  using dictionary_view_type = dictionary_type;

  template <typename T, typename IdxT>
  struct apply : storage_spec<IdxT> {
    /* The dataset stores codes, but it still represents vectors of `T`. */
    using value_type = std::remove_cv_t<T>;
    using code_type  = int8_t;
    /* NOTE: re-exported rather than declared here, so that they do not belong to one instantiation
    of `apply`; qualified lookup into `apply` does not reach the enclosing spec on its own.
    */
    using math_type            = sq_spec::math_type;
    using dictionary_type      = sq_spec::dictionary_type;
    using dictionary_view_type = sq_spec::dictionary_view_type;

    /* NOTE: every accessor is inherited; the dimension is still one code per component, so even
    `get_dim` is the dense one.
    */
    [[nodiscard]] static auto get_dictionary_view(const dictionary_type& dict) noexcept
      -> dictionary_view_type
    {
      return dict;
    }
    /* NOTE: the dictionary is its own view here, read-only or not, so one overload serves both the
    owning dataset and the view.
    */
    [[nodiscard]] static auto get_dictionary_const_view(const dictionary_type& dict) noexcept
      -> dictionary_view_type
    {
      return dict;
    }
  };
};

/************************************************************************************************
 * Convenience aliases
 ************************************************************************************************/

// Empty
template <typename T, typename IdxT>
using empty_dataset = dataset<T, IdxT, empty_spec>;

// Dense
template <typename T, typename IdxT, typename ContainerPolicy>
using contiguous_dataset =
  dataset<T, IdxT, mdarray_spec<raft::layout_c_contiguous, ContainerPolicy>>;
template <typename T, typename IdxT, typename ContainerPolicy>
using padded_dataset = dataset<T, IdxT, mdarray_spec<raft::layout_left_padded<T>, ContainerPolicy>>;
template <typename T, typename IdxT>
using device_contiguous_dataset =
  contiguous_dataset<T, IdxT, raft::device_accessor<raft::device_container_policy<T>>>;
template <typename T, typename IdxT>
using device_padded_dataset =
  padded_dataset<T, IdxT, raft::device_accessor<raft::device_container_policy<T>>>;
template <typename T, typename IdxT>
using host_contiguous_dataset =
  contiguous_dataset<T, IdxT, raft::host_accessor<raft::host_container_policy<T>>>;
template <typename T, typename IdxT>
using host_padded_dataset =
  padded_dataset<T, IdxT, raft::host_accessor<raft::host_container_policy<T>>>;

// Sparse
template <typename T, typename IdxT, typename ContainerPolicy>
using csr_dataset = dataset<T, IdxT, sparse_spec<csr_layout, ContainerPolicy>>;
template <typename T, typename IdxT, typename ContainerPolicy>
using coo_dataset = dataset<T, IdxT, sparse_spec<coo_layout, ContainerPolicy>>;
template <typename T, typename IdxT>
using device_csr_dataset =
  csr_dataset<T, IdxT, raft::device_accessor<raft::device_container_policy<T>>>;
template <typename T, typename IdxT>
using device_coo_dataset =
  coo_dataset<T, IdxT, raft::device_accessor<raft::device_container_policy<T>>>;
template <typename T, typename IdxT>
using host_csr_dataset = csr_dataset<T, IdxT, raft::host_accessor<raft::host_container_policy<T>>>;
template <typename T, typename IdxT>
using host_coo_dataset = coo_dataset<T, IdxT, raft::host_accessor<raft::host_container_policy<T>>>;

// Compressed
template <typename T, typename IdxT, typename MathT, typename BookPolicy, typename CodePolicy>
using vpq_dataset = dataset<T, IdxT, vpq_spec<MathT, BookPolicy, CodePolicy>>;
template <typename T, typename IdxT, typename MathT, typename CodePolicy>
using sq_dataset = dataset<T, IdxT, sq_spec<MathT, CodePolicy>>;
template <typename T, typename IdxT, typename MathT>
using device_vpq_dataset =
  vpq_dataset<T,
              IdxT,
              MathT,
              raft::device_accessor<raft::device_container_policy<MathT>>,
              raft::device_accessor<raft::device_container_policy<uint8_t>>>;
template <typename T, typename IdxT, typename MathT>
using host_vpq_dataset = vpq_dataset<T,
                                     IdxT,
                                     MathT,
                                     raft::host_accessor<raft::host_container_policy<MathT>>,
                                     raft::host_accessor<raft::host_container_policy<uint8_t>>>;
template <typename T, typename IdxT, typename MathT>
using device_sq_dataset =
  sq_dataset<T, IdxT, MathT, raft::device_accessor<raft::device_container_policy<int8_t>>>;
template <typename T, typename IdxT, typename MathT>
using host_sq_dataset =
  sq_dataset<T, IdxT, MathT, raft::host_accessor<raft::host_container_policy<int8_t>>>;

}  // namespace CUVS_EXPORT cuvs
