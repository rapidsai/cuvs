/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::neighbors::ivf_flat::detail {

template <typename DataTag,
          typename AccTag,
          typename IdxTag,
          int Capacity,
          int Veclen,
          bool Ascending,
          bool ComputeNorm>
struct fragment_tag_interleaved_scan {};

template <int Veclen, typename DataTag, typename AccTag, typename MetricTag>
struct fragment_tag_metric {};

template <typename IvfSampleFilterTag>
struct fragment_tag_filter {};

template <typename PostLambdaTag>
struct fragment_tag_post_lambda {};

}  // namespace cuvs::neighbors::ivf_flat::detail
