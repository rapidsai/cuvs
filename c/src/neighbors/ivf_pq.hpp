/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuvs/neighbors/ivf_pq.h>
#include <cuvs/neighbors/ivf_pq.hpp>

#include "../core/exceptions.hpp"

namespace cuvs::neighbors::ivf_pq {
/// Converts a cuvsIvfPqIndexParams struct (c) to a ivf_pq::index_params (C++) struct
void convert_c_index_params(cuvsIvfPqIndexParams params,
                            cuvs::neighbors::ivf_pq::index_params* out);

/// Converts search params from C struct to C++ struct
void convert_c_search_params(cuvsIvfPqSearchParams params,
                             cuvs::neighbors::ivf_pq::search_params* out);
}  // namespace cuvs::neighbors::ivf_pq
