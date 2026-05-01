/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::neighbors::ivf_sq::detail {

struct tag_metric_l2 {};
struct tag_metric_ip {};
struct tag_metric_cosine {};

template <typename MetricTag, int Capacity>
struct fragment_tag_ivf_sq_scan {};

template <typename FilterTag>
struct fragment_tag_ivf_sq_filter {};

}  // namespace cuvs::neighbors::ivf_sq::detail
