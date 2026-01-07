/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cuvs_mg_cagra_wrapper.h"

namespace cuvs::bench {
template class cuvs_mg_cagra<float, uint32_t>;
template class cuvs_mg_cagra<half, uint32_t>;
template class cuvs_mg_cagra<uint8_t, uint32_t>;
template class cuvs_mg_cagra<int8_t, uint32_t>;
}  // namespace cuvs::bench
