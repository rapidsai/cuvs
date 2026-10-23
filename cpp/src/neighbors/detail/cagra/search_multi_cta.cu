/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "search_multi_cta.cuh"
#include <cuvs/core/cuda_fp16.hpp>

namespace cuvs::neighbors::cagra::detail::multi_cta_search {

template struct search<float,
                       uint32_t,
                       float,
                       cuvs::neighbors::filtering::none_sample_filter,
                       uint32_t,
                       uint32_t>;

template struct search<__half,
                       uint32_t,
                       float,
                       cuvs::neighbors::filtering::none_sample_filter,
                       uint32_t,
                       uint32_t>;

template struct search<int8_t,
                       uint32_t,
                       float,
                       cuvs::neighbors::filtering::none_sample_filter,
                       uint32_t,
                       uint32_t>;

template struct search<uint8_t,
                       uint32_t,
                       float,
                       cuvs::neighbors::filtering::none_sample_filter,
                       uint32_t,
                       uint32_t>;

}  // namespace cuvs::neighbors::cagra::detail::multi_cta_search
