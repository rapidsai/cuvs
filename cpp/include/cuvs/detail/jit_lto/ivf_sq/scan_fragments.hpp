/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::neighbors::ivf_sq::detail {

struct tag_metric_l2 {};
struct tag_metric_ip {};
struct tag_metric_cosine {};

// Scan entrypoint fragment. Templated only on (Capacity, Ascending). The
// metric specialization lives in the four device-function fragments below.
template <int Capacity, bool Ascending>
struct fragment_tag_ivf_sq_scan {};

template <typename FilterTag>
struct fragment_tag_ivf_sq_filter {};

// Metric-specific device-function fragments composed in at JIT-link time.
template <typename MetricTag>
struct fragment_tag_setup_invariant_smem {};

template <typename MetricTag>
struct fragment_tag_setup_per_probe_smem {};

template <typename MetricTag>
struct fragment_tag_accumulate_distance {};

template <typename MetricTag>
struct fragment_tag_finalize_distance {};

}  // namespace cuvs::neighbors::ivf_sq::detail
