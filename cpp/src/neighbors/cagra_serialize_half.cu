/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cagra_serialize.cuh"

#include <cuda_fp16.h>

namespace cuvs::neighbors::cagra {

CUVS_INST_CAGRA_SERIALIZE(half);

}  // namespace cuvs::neighbors::cagra
