/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ivf_flat_interleaved_scan_explicit_inst.cuh"

namespace cuvs::neighbors::ivf_flat::detail {

CUVS_INST_IVF_FLAT_INTERLEAVED_SCAN(uint8_t,
                                    int64_t,
                                    filtering::bitset_filter<uint32_t COMMA int64_t>);

}  // namespace cuvs::neighbors::ivf_flat::detail
