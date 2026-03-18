/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/FragmentEntry.hpp>

namespace cuvs::neighbors::ivf_flat::detail {

template <typename DataTag,
          typename AccTag,
          typename IdxTag,
          int Capacity,
          int Veclen,
          bool Ascending,
          bool ComputeNorm>
struct InterleavedScanFragmentEntry final
  : StaticFatbinFragmentEntry<InterleavedScanFragmentEntry<DataTag,
                                                           AccTag,
                                                           IdxTag,
                                                           Capacity,
                                                           Veclen,
                                                           Ascending,
                                                           ComputeNorm>> {
  static const uint8_t* const data;
  static const size_t length;
};

template <int Veclen, typename DataTag, typename AccTag, typename MetricTag>
struct MetricFragmentEntry final
  : StaticFatbinFragmentEntry<MetricFragmentEntry<Veclen, DataTag, AccTag, MetricTag>> {
  static const uint8_t* const data;
  static const size_t length;
};

template <typename PostLambdaTag>
struct PostLambdaFragmentEntry final
  : StaticFatbinFragmentEntry<PostLambdaFragmentEntry<PostLambdaTag>> {
  static const uint8_t* const data;
  static const size_t length;
};

}  // namespace cuvs::neighbors::ivf_flat::detail
