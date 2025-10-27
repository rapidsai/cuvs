/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cuvs_mg_ivf_flat_wrapper.h"

namespace cuvs::bench {
template class cuvs_mg_ivf_flat<float, int64_t>;
// template class cuvs_mg_ivf_flat<half, int64_t>;
template class cuvs_mg_ivf_flat<uint8_t, int64_t>;
template class cuvs_mg_ivf_flat<int8_t, int64_t>;
}  // namespace cuvs::bench
