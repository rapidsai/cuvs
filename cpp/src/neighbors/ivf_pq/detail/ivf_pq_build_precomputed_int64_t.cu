/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/ivf_pq.hpp>

#include "ivf_pq_build_precomputed_inst.cuh"

namespace cuvs::neighbors::ivf_pq {
CUVS_INST_IVF_PQ_BUILD_PRECOMPUTED(int64_t);

#undef CUVS_INST_IVF_PQ_BUILD_PRECOMPUTED

}  // namespace cuvs::neighbors::ivf_pq

