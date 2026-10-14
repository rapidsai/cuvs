/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/dataset/view.hpp>
#include <cuvs/core/export.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace CUVS_EXPORT cuvs {

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

  /* The former name of the read-only `view()`, kept so that the existing callers keep working. */
  [[nodiscard]] auto as_dataset_view() const noexcept -> const_view_type { return view(); }

  [[nodiscard]] auto n_rows() const noexcept -> index_type { return spec_type::get_n_rows(data_); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    // NOTE: a compressed dataset may not be able to tell its dimension from the encoded data alone
    return spec_type::get_dim(data_, dictionary_);
  }
  /* The distance between the starts of two rows, in elements; defined if the spec has strided rows.
   */
  [[nodiscard]] auto stride() const noexcept -> uint32_t
    requires requires(const data_type& d) { spec_type::get_stride(d); }
  {
    return spec_type::get_stride(data_);
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
  [[nodiscard]] auto dictionary_view() noexcept -> dictionary_view_type
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

}  // namespace CUVS_EXPORT cuvs
