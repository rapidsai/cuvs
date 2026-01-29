/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ivf_flat_interleaved_scan_explicit_inst.cuh"

namespace cuvs::neighbors::ivf_flat::detail {

CUVS_INST_IVF_FLAT_INTERLEAVED_SCAN(float, int64_t, filtering::none_sample_filter);
}  // namespace cuvs::neighbors::ivf_flat::detail
