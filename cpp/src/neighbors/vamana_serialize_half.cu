/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_fp16.h>
#include "vamana_serialize.cuh"

namespace cuvs::neighbors::vamana {

CUVS_INST_VAMANA_SERIALIZE(half);

}  // namespace cuvs::neighbors::vamana
