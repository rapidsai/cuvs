/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cuvs_mg_ivf_pq_wrapper.h"

namespace cuvs::bench {
template class cuvs_mg_ivf_pq<float, int64_t>;
template class cuvs_mg_ivf_pq<half, int64_t>;
template class cuvs_mg_ivf_pq<uint8_t, int64_t>;
template class cuvs_mg_ivf_pq<int8_t, int64_t>;
}  // namespace cuvs::bench
