/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cuvs_cagra_wrapper.h"

namespace cuvs::bench {
template class cuvs_cagra<uint8_t, uint32_t>;
}  // namespace cuvs::bench
