/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/FragmentEntry.hpp>

namespace cuvs::neighbors::ivf_pq::detail {

struct tag_out_f {};
struct tag_out_h {};

struct tag_lut_f {};
struct tag_lut_h {};
struct tag_lut_fp8_signed {};
struct tag_lut_fp8_unsigned {};

template <typename OutTag, typename LutTag>
struct ComputeSimilarityFragmentEntry final
  : StaticFatbinFragmentEntry<ComputeSimilarityFragmentEntry<OutTag, LutTag>> {
  static const uint8_t* const data;
  static const size_t length;
};

template <typename LutTag, bool EnableSMemLut, uint32_t PqBits>
struct PrepareLutFragmentEntry final
  : StaticFatbinFragmentEntry<PrepareLutFragmentEntry<LutTag, EnableSMemLut, PqBits>> {
  static const uint8_t* const data;
  static const size_t length;
};

template <typename OutTag, bool kManageLocalTopK>
struct StoreCalculatedDistancesFragmentEntry final
  : StaticFatbinFragmentEntry<StoreCalculatedDistancesFragmentEntry<OutTag, kManageLocalTopK>> {
  static const uint8_t* const data;
  static const size_t length;
};

template <bool PrecompBaseDiff>
struct PrecomputeBaseDiffFragmentEntry final
  : StaticFatbinFragmentEntry<PrecomputeBaseDiffFragmentEntry<PrecompBaseDiff>> {
  static const uint8_t* const data;
  static const size_t length;
};

template <typename LutTag, bool PrecompBaseDiff, uint32_t PqBits>
struct CreateLutFragmentEntry final
  : StaticFatbinFragmentEntry<CreateLutFragmentEntry<LutTag, PrecompBaseDiff, PqBits>> {
  static const uint8_t* const data;
  static const size_t length;
};

template <typename OutTag, typename LutTag, int Capacity, uint32_t PqBits>
struct ComputeDistancesFragmentEntry final
  : StaticFatbinFragmentEntry<ComputeDistancesFragmentEntry<OutTag, LutTag, Capacity, PqBits>> {
  static const uint8_t* const data;
  static const size_t length;
};

}  // namespace cuvs::neighbors::ivf_pq::detail
